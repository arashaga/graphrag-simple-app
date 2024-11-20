# indexing.py

import asyncio
import time
from pathlib import Path
import os
from dotenv import load_dotenv
import nest_asyncio


# **Set the event loop policy for Windows at module level**
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# **Apply nest_asyncio to allow nested event loops**
nest_asyncio.apply() 

async def run_build_index():
    from graphrag.index.api import build_index
    from graphrag.config import load_config, resolve_paths, enable_logging_with_config
    from graphrag.index.progress.load_progress_reporter import load_progress_reporter
    from graphrag.index.progress.types import ReporterType
    # Set the root directory to the path where your data and settings.yaml are located
    root_dir = '../'
    root = Path(root_dir).resolve()
    print("root directory: ",root)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_directory, '../../backend/.env')

    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_path)

    # Alternatively, set environment variables directly
    # os.environ['GRAPHRAG_API_KEY'] = 'your_api_key_here'
    # Load the configuration from settings.yaml
    config = load_config(root, config_filepath=None)

    # Set a unique run ID
    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Resolve paths based on the configuration and run ID
    resolve_paths(config, run_id)

    # Enable logging as per the configuration
    enable_logging_with_config(config, verbose=True)

    # Create a progress reporter (options: 'silent', 'rich')
    progress_reporter = load_progress_reporter(ReporterType.PRINT)



    # Run build_index
    outputs = await build_index(
        config=config,
        run_id=run_id,
        is_resume_run=False,
        is_update_run=False,
        memory_profile=False,
        progress_reporter=progress_reporter,
        emit=[],
    )

    # Handle outputs and check for errors
    encountered_errors = any(
        output.errors and len(output.errors) > 0 for output in outputs
    )

    if encountered_errors:
        print("Errors occurred during the indexing process.")
        raise Exception("Indexing failed due to errors.")
    else:
        print("Indexing completed successfully.")

def run_indexing():
    """
    Synchronously run the asynchronous build_index function.
    """
    try:
        asyncio.run(run_build_index())
    except Exception as e:
        print(f"Error running build_index: {e}")
        raise