#!/usr/bin/env python3
"""CLI for ICL Workflow."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workflow import ICLWorkflow, WorkflowConfig
from mcp_server import ICLWorkflowServer


def run_workflow():
    """Run complete ICL workflow from command line."""
    parser = argparse.ArgumentParser(description="ICL Subtraction Workflow")
    parser.add_argument("image", help="Input FITS image path")
    parser.add_argument("--instrument", "-i", required=True,
                       choices=["CSST", "Euclid", "HST", "JWST"],
                       help="Telescope instrument")
    parser.add_argument("--center-x", "-x", type=float,
                       help="Lens galaxy center X (auto-detected if not provided)")
    parser.add_argument("--center-y", "-y", type=float,
                       help="Lens galaxy center Y (auto-detected if not provided)")
    parser.add_argument("--mode", "-m", choices=["auto", "semi_auto", "manual"],
                       help="Workflow mode (auto-detected if not provided)")
    parser.add_argument("--output", "-o", default="./outputs",
                       help="Output directory")
    parser.add_argument("--detect-thresh", type=float, default=1.5,
                       help="Detection threshold (default: 1.5)")
    parser.add_argument("--dilation", "-d", type=float, default=1.5,
                       help="Dilation factor (default: 1.5)")
    parser.add_argument("--method", choices=["rbf", "clough-tocher", "linear", "nearest"],
                       default="rbf", help="Interpolation method")

    args = parser.parse_args()

    # Configure workflow
    config = WorkflowConfig(
        output_dir=args.output,
        detect_thresh=args.detect_thresh,
        dilation_factor=args.dilation,
        interpolation_method=args.method,
    )

    # Create and run workflow
    workflow = ICLWorkflow(config)

    # Register simple HITL callback
    def human_callback(step_name: str, result) -> bool:
        print(f"\n{'='*60}")
        print(f"HUMAN INPUT REQUIRED: {step_name}")
        print(f"{'='*60}")
        print(f"Message: {result.message}")
        if result.visualization_path:
            print(f"Visualization: {result.visualization_path}")
        if result.metrics:
            print(f"Metrics: {result.metrics}")

        while True:
            response = input("\nApprove? [y/n]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")

    workflow.register_human_callback(human_callback)

    # Run workflow
    final_state = workflow.run(
        args.image,
        args.instrument,
        center_x=args.center_x,
        center_y=args.center_y,
        mode=args.mode
    )

    # Print summary
    print("\n" + workflow.get_summary())


def run_server():
    """Run MCP server."""
    parser = argparse.ArgumentParser(description="ICL Workflow MCP Server")
    parser.add_argument("--output", "-o", default="./outputs",
                       help="Output directory")
    parser.add_argument("--stdio", action="store_true",
                       help="Use stdio transport (for MCP clients)")

    args = parser.parse_args()

    config = WorkflowConfig(output_dir=args.output)
    server = ICLWorkflowServer(config)

    if args.stdio:
        # MCP stdio transport
        import json
        import sys

        print("ICL Workflow MCP Server running on stdio", file=sys.stderr)

        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = server.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }), flush=True)
    else:
        # Simple interactive mode
        print("ICL Workflow MCP Server")
        print(f"Capabilities: {json.dumps(server.get_capabilities(), indent=2)}")
        print("\nServer ready. Use --stdio for MCP client integration.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ICL Workflow Tools")
    parser.add_argument("command", choices=["workflow", "server", "tools"],
                       help="Command to run")

    # Parse known args to pass remaining to subcommand
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining

    if args.command == "workflow":
        run_workflow()
    elif args.command == "server":
        run_server()
    elif args.command == "tools":
        # List available tools
        server = ICLWorkflowServer()
        caps = server.get_capabilities()
        print("Available Tools:")
        for tool in caps["tools"]:
            print(f"  - {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    main()
