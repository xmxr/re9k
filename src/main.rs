extern crate petgraph;
extern crate r2pipe;
extern crate serde_json;

use r2pipe::R2Pipe;
use regex::Regex;
use std::collections::{HashMap, HashSet};
//use petgraph::graph::Graph;
use petgraph::graphmap::GraphMap;
use clap::Parser;

use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::backend::Autodiff;

use re9k::training;
use re9k::inference;

const FUNS: [&str; 6] = [
    "madvise",
    "prctl",
    "signal",
    "sigaction",
    "process_vm_writev",
    "ptrace",
];
const STR: [(&str, &str); 12] = [
    (r"signal[\s]*\([0x]*5,", "SIGNAL_SIGTRAP"),
    (r"sigaction[\s]*\([0x]*5,", "SIGACT_SIGTRAP"),
    (r"ptrace[\s]*\([0x]*0,", "PTRACE_TRACEME"),
    (r"ptrace[\s]*\([0x]*1,", "PTRACE_PEEKTEXT"),
    (r"ptrace[\s]*\([0x]*4,", "PTRACE_POKETEXT"),
    (r"ptrace[\s]*\([0x]*2,", "PTRACE_PEEKDATA"),
    (r"ptrace[\s]*\([0x]*5,", "PTRACE_POKEDATA"),
    (r"ptrace[\s]*\(0x10,", "PTRACE_ATTACH"),
    (r"ptrace[\s]*\(0x4206,", "PTRACE_SEIZE"),
    (r"prctl[\s]*\([0x]*4,", "PR_SET_DUMPABLE"),
    (r"prctl[\s]*\(0xf,", "PR_SET_NAME"),
    (r"madvise[^\\n]*, 0x10\)", "MADV_DONTDUMP"),
];

#[derive(Debug)]
struct Sample<'a> {
    name: String,
    arch: String,
    bits: u64,
    compiler: String,
    stripped: bool,
    link_static: bool,
    sect_header: bool,
    functions: Vec<String>,
    optimized: u32,
    params: HashSet<&'a str>,
    cff: Vec<String>
}

#[repr(usize)]
#[derive(Debug, PartialEq)]
enum JumpType {
    Unconditional = 0,
    Conditional = 1,
}

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    file: Option<String>,
    #[arg(short, long)]
    train: Option<String>
}


fn find_imports(s: &mut Sample, r2: &mut R2Pipe) {
    let bind = r2.cmdj("isj").expect("Couldn't fetch imports");
    let imp = bind.as_array().unwrap();
    //let imp_funs: Vec<&str> = imp.into_iter().map(|f| f["realname"].as_str().unwrap()).collect();
    let imp_funs: Vec<&str> = imp
        .into_iter()
        .map(|f| f["flagname"].as_str().unwrap())
        .collect();

    let matches = imp_funs
        .iter()
        //.filter(|&imp_fun| FUNS.iter().any(|&fun| *fun == **imp_fun))
        .filter(|imp_fun| FUNS.iter().any(|fun| imp_fun.contains(fun)))
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    s.functions = matches;
}

fn find_links(s: &mut Sample, r2: &mut R2Pipe) {
    let bind = r2.cmdj("aflj").expect("Couldn't fetch functions");
    let link_funs: Vec<&str> = bind
        .as_array()
        .unwrap()
        .into_iter()
        .map(|f| f["name"].as_str().unwrap())
        .collect();

    let matches = link_funs
        .iter()
        .filter(|link_fun| FUNS.iter().any(|&fun| link_fun.ends_with(fun)))
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    s.functions = matches;
}

fn find_strip(s: &mut Sample, r2: &mut R2Pipe) {
    let bind = r2.cmdj("/asj").expect("Couldn't fetch syscalls");
    let syscalls = bind["results"]
        .as_array()
        .unwrap()
        .into_iter()
        .filter(|sys| {
            FUNS.iter().any(|&fun| {
                let s = sys["name"].as_str().unwrap();
                !s.starts_with("arch") && s.contains(fun)
            })
        })
        .collect::<Vec<_>>();

    let mut matches = vec![];

    for sys in syscalls {
        let fcn = r2
            .cmdj(format!("afdj @ {}", sys["addr"]).as_str())
            .expect("Coulnd't find function");

        let fcn_name = fcn["name"].as_str().unwrap();
        let sys_name = sys["name"].as_str().unwrap();

        let rename = format!("{fcn_name}_{sys_name}");
        r2.cmd(format!("afn {rename} {fcn_name}").as_str())
            .expect("Renaming failed");

        matches.push(rename.clone());

        //sigaction -> signal
        //signal shouldnt contain any syscalls
        if sys_name.contains("sigaction") {
            let bind = r2
                .cmdj(format!("axtj @ {rename}").as_str())
                .expect("Couldn't fetch sigaction callers");

            let mut sig_callers = bind.as_array().unwrap().iter().collect::<Vec<_>>();

            sig_callers.sort_by_key(|x| x["fcn_addr"].as_u64().unwrap());
            sig_callers.dedup_by_key(|x| x["fcn_addr"].as_u64().unwrap());

            for sig_call in sig_callers {
                //for sig_call in bind.as_array().unwrap() {
                let signal_fcn = sig_call["fcn_name"].as_str().unwrap();
                let disas = r2
                    .cmd(format!("pif @ {} ~[0]", signal_fcn).as_str())
                    .unwrap();

                match (disas.contains("svc"), disas.contains("syscall")) {
                    (false, false) => {
                        let rename = format!("{signal_fcn}_signal");
                        r2.cmd(format!("afn {rename} {signal_fcn}").as_str())
                            .expect("Signal rename failed");
                        matches.push(rename);
                    }
                    _ => (),
                }
            }
        }
    }
    s.functions = matches;
}

fn check_funs(s: &mut Sample, r2: &mut R2Pipe) {
    s.functions = s
        .functions
        .iter()
        .filter(|fun| {
            let res = r2.cmd(format!("axg @ {fun} ~entry0").as_str()).unwrap();
            res.len() > 0
        })
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let mut reg_map = HashMap::new();
    for (reg, tag) in STR {
        reg_map.insert(tag, Regex::new(reg).unwrap());
    }

    for fcn in &s.functions {
        let bind = r2
            .cmdj(format!("axtj @ {fcn}").as_str())
            .expect("Calls to fun");
        let calls = bind.as_array().unwrap();

        for fcn_call in calls {
            let Some(fcn_name) = fcn_call["fcn_name"].as_str() else {
                continue;
            };

            for dec in ["pdc", "pdg"] {
                let decomp = r2
                    .cmd(format!("{dec} @ {fcn_name}").as_str())
                    .expect("decomp");
                for (tag, reg) in &reg_map {
                    if reg.is_match(&decomp) {
                        s.params.insert(tag);
                    }
                }
            }
        }
    }
}

fn check_flat_cfg(fun_vec: &Vec<&serde_json::Value>, s: &mut Sample, r2: &mut R2Pipe) {
    'x: for fun in fun_vec.into_iter().rev().take(25) {
        let Some(fcn_name) = fun["name"].as_str() else {
            continue;
        };
        let bind = r2
            .cmd(format!("agfm @ {fcn_name}").as_str())
            .expect("graph");
        let data: Vec<&str> = bind.split("\n").collect();

        let mut graph = GraphMap::<&str, JumpType, petgraph::Directed>::new();
        data.iter().for_each(|x| match x.find("-->") {
            None => (),
            Some(idx) => {
                let src = x[0..idx].trim();
                let tmp_dst = &x[idx + 3..];

                let (dst, edge) = match tmp_dst.find(":") {
                    None => (tmp_dst.trim(), JumpType::Unconditional),
                    Some(idx) => {
                        let (dst, _) = tmp_dst.split_at(idx);
                        (dst.trim(), JumpType::Conditional)
                    }
                };
                graph.add_edge(src, dst, edge);
            }
        });

        let Some(max_node) = graph
            .nodes()
            .max_by_key(|n| {
                graph
                    .edges_directed(n, petgraph::Direction::Incoming)
                    .count()
            }) else {
            continue;
        };

        for (_,_,e) in graph.edges_directed(max_node, petgraph::Direction::Incoming) {
            if *e == JumpType::Conditional {
                continue 'x;
            }
        }

        for dispatch in graph.nodes().take(5) {
            if let Some(JumpType::Unconditional) = graph.edge_weight(max_node, dispatch) {
                s.cff.push(fcn_name.to_string());
            }
        }
    }
}

fn infer_opt(fun_vec: &Vec<&serde_json::Value>, s: &mut Sample, r2: &mut R2Pipe) {
    let devices = vec![LibTorchDevice::Cpu];
    let mut label = vec![];

    for fun in fun_vec.into_iter().rev().take(25) {
        let Some(fcn_name) = fun["name"].as_str() else {
            continue;
        };
        let fun_disas = r2
            .cmd(format!("pif @ {fcn_name} ~[0]").as_str())
            .expect("function disas")
            .replace('\n', " ");

        label.push(inference::infer::<Autodiff<LibTorch>>(devices.clone(), fun_disas));
    }

    s.optimized = 100 * label.iter().sum::<u32>() / label.len() as u32;
}

fn inspect(file: &str) {
    let mut r2 = R2Pipe::spawn(file, None).unwrap();

    assert_ne!(r2.cmd("afi entry0").unwrap(), "\n".to_string());
    r2.cmd("aaa").expect("Analysis failed");

    let data = r2.cmdj("ij").unwrap();
    let mut sample = Sample {
        name: data["core"]["file"].to_string(),
        arch: data["bin"]["arch"].to_string(),
        bits: data["bin"]["bits"].as_u64().unwrap(),
        compiler: data["bin"]["compiler"].to_string(),
        stripped: data["bin"]["stripped"].as_bool().unwrap(),
        link_static: data["bin"]["static"].as_bool().unwrap(),
        sect_header: r2
            .cmdj("iSj")
            .expect("fetch sections")
            .as_array()
            .unwrap()
            .len()
            > 3,
        functions: vec![],
        optimized: 0,
        params: HashSet::new(),
        cff: vec![]
    };

    match (sample.link_static, sample.stripped) {
        (false, _) => find_imports(&mut sample, &mut r2),
        (true, false) => find_links(&mut sample, &mut r2),
        (true, true) => find_strip(&mut sample, &mut r2),
    }

    let bind = r2.cmdj("aflj").expect("Fetch function list");
    let mut fun_vec = bind
        .as_array()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    fun_vec.sort_by_key(|x| x["cc"].as_u64().unwrap()); //sort by cyclomatic complexity
    
    check_funs(&mut sample, &mut r2);
    check_flat_cfg(&fun_vec, &mut sample, &mut r2);
    infer_opt(&fun_vec, &mut sample, &mut r2);
    println!("{:?}", sample);

    r2.close();
}

fn main() {
    let cli = Cli::parse();
    if let Some(file) = cli.file.as_deref() {
        inspect(file);
    }
    if let Some(ds_train) = cli.train.as_deref() {
        let devices = vec![LibTorchDevice::Cpu];
        training::run::<Autodiff<LibTorch>>(devices, ds_train);
    }
}
