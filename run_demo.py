"""One entry point, independently evaluated modules."""
import argparse, json, pathlib
from atl_demo import attribution, federated, probe

def main():
    p=argparse.ArgumentParser(description="ATL ECHO public Qwen research reference demo")
    p.add_argument("--module",choices=["all","a","b","c"],default="all")
    p.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--scenarios",type=int,default=5)
    p.add_argument("--skip-qwen",action="store_true",help="Run only lightweight evidence/masking checks")
    p.add_argument("--output",default="results/demo_results.json")
    args=p.parse_args(); results=[]
    if args.module in ("all","a"): results.append(attribution.run())
    if args.module in ("all","b") and not args.skip_qwen:
        try: results.append(probe.run(args.model,args.scenarios))
        except (ImportError,OSError,RuntimeError) as e:
            results.append({"module":"B: Qwen open linear probe","status":"not run", "reason":f"{type(e).__name__}: {str(e)[:400]}","warning":"Install dependencies and run with adequate memory/GPU. Do not report this as a successful experiment."})
    if args.module in ("all","c"): results.append(federated.run())
    output=pathlib.Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2,ensure_ascii=False));print(f"\nSaved: {output}")

if __name__=="__main__": main()
