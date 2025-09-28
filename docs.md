flowchart TD
    A([start]) --> B[update_task_state: 'Processing user query']
    B --> C[preprocess_query(query)\n→ LLMProcessedQuery]
    C --> D[find_relevant_papers(llm_processed_query)\n→ snippets + search_api]
    D --> E{retrieved_candidates rỗng?}
    E -- yes --> E1[[raise Exception\n'no relevant information']]
    E -- no --> F[rerank_and_aggregate(query, candidates)\n→ reranked_df + paper_metadata]
    F --> G{reranked_df.empty?}
    G -- yes --> G1[[raise Exception\n'No relevant papers post reranking']]
    G -- no --> H[step_select_quotes(query, reranked_df)\n→ per_paper_summaries]
    H --> I{per_paper_summaries rỗng?}
    I -- yes --> I1[[raise Exception\n'No relevant quotes extracted']]
    I -- no --> J[step_clustering(query, per_paper_summaries)\n→ cluster_json]
    J --> K[build plan_json từ cluster_json.dimensions]
    K --> L{plan_json có quote index?}
    L -- no --> L1[[raise Exception\n'planning failed']]
    L -- yes --> M[extract_quote_citations(reranked_df,\nper_paper_summaries, plan_json,\npaper_metadata)\n→ per_paper_summaries_extd + quotes_metadata]
    M --> N[update_task_state: 'Start generating sections']
    N --> O[step_gen_iterative_summary(query,\nper_paper_summaries_extd, plan_json)\n→ generator]
    O --> P{{loop từng section}}
    P --> Q[get_json_summary(...)\n→ section_json]
    Q --> R[postprocess_json_output(json_summary,\nquotes_meta)]
    R --> S{section_json.format == 'list'\n& run_table_generation?}
    S -- yes --> T[gen_table_thread(...)\n→ spawn TableGenerator thread]
    S -- no --> U[append GeneratedSection]
    T --> U[append GeneratedSection]
    U --> P
    P -- done --> V[update_task_state: 'Generating comparison tables']
    V --> W[join tất cả table threads]
    W --> X[attach tables vào json_summary & sections]
    X --> Y[postprocess_json_output(...)]
    Y --> Z[trace & persist EventTrace]
    Z --> AA([return TaskResult(sections, cost, tokens)])
