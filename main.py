import argparse
import json
import sys

from harness.config import list_servers, load_server_config
from harness.lifecycle import setup_server, teardown_server
from harness.runner import print_table, run, save_results


def cmd_run(args):
    from harness.providers import detect_default_model

    config = load_server_config(args.server)
    models = [args.model] if args.model else config["models"]
    if not models:
        default = detect_default_model()
        print(f"No model specified — auto-detected: {default}")
        models = [default]

    repeats = args.repeats or config["repeats"]
    threshold = args.threshold or config["threshold"]
    metric_names = [m.strip() for m in args.metrics.split(",")] if args.metrics else config["metrics"]

    process = setup_server(config)
    failed = False

    try:
        for model in models:
            print(f"\n{'='*60}")
            print(f"Server: {config['name']}  |  Model: {model}")
            print(f"{'='*60}\n")

            run_data = run(
                config["tool_source"],
                config["test_cases"],
                model,
                repeats=repeats,
                metric_names=metric_names,
                threshold=threshold,
                judge_model=args.judge_model,
            )

            if args.output == "json":
                print(json.dumps(run_data, indent=2))
            else:
                print_table(run_data)

            path = save_results(run_data, config["name"])
            print(f"\nSaved results to {path}")

            if args.ci and run_data["accuracy"] < threshold:
                failed = True
    finally:
        teardown_server(process, config)

    if failed:
        print(f"\nCI FAILED: one or more models below threshold {threshold:.1%}")
        sys.exit(1)


def cmd_run_all(args):
    servers = list_servers()
    if not servers:
        print("No servers registered in servers/", file=sys.stderr)
        sys.exit(1)

    print(f"Running evals for {len(servers)} servers: {', '.join(servers)}\n")

    failed = []
    for server_name in servers:
        args.server = server_name
        try:
            cmd_run(args)
        except SystemExit as e:
            if e.code == 1:
                failed.append(server_name)
        except Exception as e:
            print(f"\nError running {server_name}: {e}", file=sys.stderr)
            failed.append(server_name)

    if failed:
        print(f"\nFailed servers: {', '.join(failed)}")
        sys.exit(1)


def cmd_list(args):
    from harness.config import load_raw_config
    from harness.providers import detect_available_providers, detect_default_model

    servers = list_servers()
    if not servers:
        print("No servers registered in servers/")
        return

    available = detect_available_providers()
    if available:
        print(f"Detected providers: {', '.join(available.keys())}")
        print(f"Default model: {detect_default_model()}\n")
    else:
        print("No cloud providers detected — defaulting to local Ollama\n")

    print(f"{'NAME':<20}{'TRANSPORT':<15}{'MODELS':<40}{'TESTS':<8}")
    for name in servers:
        config = load_raw_config(name)
        models = config.get("models", [])
        models_str = ", ".join(models) if models else "(auto-detect)"
        test_count = len(config.get("test_cases", []))
        transport = config.get("transport", "unknown")
        print(f"{name:<20}{transport:<15}{models_str:<40}{test_count:<8}")


def cmd_generate(args):
    from harness.mcp_client import get_tools
    from harness.providers import call_model

    import asyncio

    config = load_server_config(args.server)
    process = setup_server(config)

    try:
        tools = asyncio.run(get_tools(config["tool_source"]))
        model = args.model or (config["models"][0] if config["models"] else None)
        if not model:
            print("Error: no model specified and none configured", file=sys.stderr)
            sys.exit(1)

        count = args.count or len(tools)

        print(f"Generating test cases for {min(count, len(tools))} tools using {model}...\n")
        print("test_cases:")

        for tool in tools[:count]:
            prompt = (
                f"Generate a natural language query that a user would type to trigger "
                f"the tool '{tool['name']}' (description: {tool['description']}). "
                f"Return ONLY the query text, nothing else."
            )
            result = call_model(model, prompt, [])
            query = result[0].name if result else f"Use {tool['name']}"

            print(f"  - id: GEN-{tool['name']}")
            print(f"    query: \"{prompt}\"")
            print(f"    expected_tool: {tool['name']}")
            print(f"    category: generated")
            print()
    finally:
        teardown_server(process, config)


def main():
    parser = argparse.ArgumentParser(description="MCP tool-calling evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared flags
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", default=None, help="Override model (provider:name format)")
    shared.add_argument("--repeats", type=int, default=None, help="Runs per test case")
    shared.add_argument("--metrics", default=None, help="Comma-separated metrics")
    shared.add_argument("--threshold", type=float, default=None, help="Pass/fail threshold 0.0-1.0")
    shared.add_argument("--judge-model", default=None, help="Model for LLM-as-judge metrics")
    shared.add_argument("--ci", action="store_true", help="Exit with code 1 if below threshold")
    shared.add_argument("--output", choices=["table", "json"], default="table", help="Output format")

    # run <server>
    run_parser = subparsers.add_parser("run", parents=[shared], help="Run evals for a server")
    run_parser.add_argument("server", help="Server name (matches servers/<name>.yaml)")

    # run-all
    subparsers.add_parser("run-all", parents=[shared], help="Run evals for all registered servers")

    # list
    subparsers.add_parser("list", help="List registered servers")

    # generate <server>
    gen_parser = subparsers.add_parser("generate", help="Generate test cases from tool discovery")
    gen_parser.add_argument("server", help="Server name")
    gen_parser.add_argument("--model", default=None, help="Model for generation")
    gen_parser.add_argument("--count", type=int, default=None, help="Max tools to generate for")

    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "run-all": cmd_run_all,
        "list": cmd_list,
        "generate": cmd_generate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
