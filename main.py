from helpers import run_research_task

def main():
    """
    Main program.
    """
    try:
        while True:
            # Get the research query from the user
            research_query = input("Please enter your research query (or type 'exit' to quit): ")

            # Check for exit command
            if research_query.lower() in ['exit', 'quit']:
                print("Exiting the program. Goodbye!")
                break

            # Run the research task only if the user provided a non-empty query
            if research_query.strip():
                run_research_task(research_query)
            else:
                print("No query provided. Please enter a valid research topic.\n")

    except KeyboardInterrupt:
        print("\nProcess interrupted by keyboard. Exiting.")


if __name__ == '__main__':
    main()