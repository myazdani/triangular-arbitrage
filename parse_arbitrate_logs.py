#!/usr/bin/env python3
# HOW TO RUN:
# python3  python3 parse_arbitrate_logs.py my_arbitrage_log.txt -o parsed.json

import re
import sys
import json
import argparse

def parse_logs(text):
    # Regex to capture each arbitrage opportunity block with variable number of steps
    pattern = re.compile(
        r"TIMESTAMP:\s*(?P<timestamp>.+?)\n"
        r"RETURN CODE:\s*(?P<return_code>\d+).*?"
        r"New\s*(?P<percent>[\d\.]+)%\s*(?P<exchange>\w+)\s*opportunity:\s*\n"
        r"(?P<steps>(?:\d+\.\s*.+?\n?)+)"
        r"(?=\n=+|\Z)",
        re.DOTALL
    )
    
    records = []
    for m in pattern.finditer(text):
        # Parse the steps section to extract individual steps
        steps_text = m.group('steps').strip()
        steps = []
        
        # Extract each numbered step
        step_pattern = re.compile(r'^\d+\.\s*(.+?)$', re.MULTILINE)
        for step_match in step_pattern.finditer(steps_text):
            steps.append(step_match.group(1).strip())
        
        records.append({
            'timestamp': m.group('timestamp').strip(),
            'return_code': int(m.group('return_code').strip()),
            'return_pct': float(m.group('percent').strip()),
            'exchange': m.group('exchange').strip(),
            'steps': steps
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Parse arbitrage logs into structured JSON and save to disk"
    )
    parser.add_argument(
        'infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
        help="Log file to parse (defaults to stdin)"
    )
    parser.add_argument(
        '-o', '--output', type=str,
        help="Path to save the output JSON file (if omitted, prints to stdout)"
    )
    args = parser.parse_args()

    text = args.infile.read()
    data = parse_logs(text)
    json_str = json.dumps(data, indent=2)

    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(json_str)
            print(f"Saved parsed output to {args.output}")
        except IOError as e:
            print(f"Error saving to {args.output}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(json_str)

if __name__ == '__main__':
    main()
