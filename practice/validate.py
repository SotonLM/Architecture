"""Hidden test validator for exercises.

This script validates student solutions without showing the test code.
Students can run this to check their work.
"""

import sys
import importlib.util
from pathlib import Path

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_problem(problem_num: int, module_name: str, description: str) -> bool:
    """Check if a problem solution passes tests."""
    print(f"\n{BOLD}Problem {problem_num}: {description}{RESET}")
    print("-" * 50)
    
    try:
        # Try to import the module
        module_path = Path(f"architecture/{module_name}.py")
        if not module_path.exists():
            print(f"{RED}✗ File not found: architecture/{module_name}.py{RESET}")
            return False
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Import and run hidden tests
        from hidden_tests import run_hidden_test
        result, message = run_hidden_test(problem_num, module)
        if result:
            print(f"{GREEN}✓ PASSED{RESET}")
            if message:
                print(f"  {message}")
            return True
        else:
            print(f"{RED}✗ FAILED{RESET}")
            if message:
                print(f"  {message}")
            return False
            
    except AssertionError as e:
        print(f"{RED}✗ FAILED: Test assertion failed{RESET}")
        print(f"  Hint: Check your implementation logic")
        return False
    except AttributeError as e:
        print(f"{RED}✗ FAILED: Missing required function or class{RESET}")
        print(f"  Error: {str(e)}")
        return False
    except Exception as e:
        print(f"{RED}✗ FAILED: {type(e).__name__}{RESET}")
        print(f"  Error: {str(e)}")
        return False


def main():
    """Run validation for all problems."""
    print(f"{BOLD}{'='*60}")
    print("  Transformer Architecture Exercise Validator")
    print(f"{'='*60}{RESET}\n")
    
    problems = [
        (1, "basics", "Basic Tensor Operations"),
        (2, "attention", "Dot Product Attention"),
        (3, "positional_encoding", "Positional Encoding"),
        (4, "attention", "Simple Attention Module"),
    ]
    
    results = []
    for problem_num, module, description in problems:
        passed = check_problem(problem_num, module, description)
        results.append((problem_num, passed))
    
    # Summary
    print(f"\n{BOLD}{'='*60}")
    print("  Summary")
    print(f"{'='*60}{RESET}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for problem_num, passed in results:
        status = f"{GREEN}✓ PASSED{RESET}" if passed else f"{RED}✗ FAILED{RESET}"
        print(f"  Problem {problem_num}: {status}")
    
    print(f"\n{BOLD}Total: {passed_count}/{total_count} problems passed{RESET}\n")
    
    if passed_count == total_count:
        print(f"{GREEN}{BOLD}Congratulations! All tests passed! 🎉{RESET}\n")
        return 0
    else:
        print(f"{YELLOW}Keep working on the failed problems. You can do it! 💪{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

