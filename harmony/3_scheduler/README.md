# Harmony Scheduler

This directory contains code to schedule Harmony tasks by consuming the profiles generated from Harmony Profiler (`2_profiler`) via following modules:

- `scheduler.py` (main): schedule Harmony in one of the two modes:
  
  - automated search (default): iteratively search for the best (minimal runtime) configured task graph via creating a four-tuple configuration by packing layers under different microbatch sizes, composing a task graph from a configuration, and estimating the runtime of a task graph. This is implemented in `search.py`.

  - manual: manually create a task graph with given microbatch sizes and constant sized layer packing. This is mostly used for understanding and debugging.
  
- `task_graph_composor.py`: compose a task graph based on a given configuration and use customized data structures in `task_data_struct.py`. 
  
- `simulator.py`: estimate runtime time of a composed task graph by performing an event-driven simulation with an optional visualization in `chrome-trace` (chrome://tracing/). It uses cutomized event data structure in `sim_data_struct.py` and helper for visualization in `sim_chrome_trace.py`.

## Example Scripts

Example scripts (`run_<model>.sh`) are provided for different models. 
Then the generated `<schedule>.pickle` (i.e., the best configured task graph) will be saved to `../results/<model>`, which will be used by downstreams of Harmony. 