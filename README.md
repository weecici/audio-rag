# Audio2Text RAG - CS431 Final Project

## Architecture Diagrams

```mermaid
%%{init: {
  "theme": "dark",
  "themeVariables": {
    "primaryColor": "#2a2f4a",
    "primaryTextColor": "#ffffff",
    "secondaryColor": "#1f2233",
    "tertiaryColor": "#30344d",
    "lineColor": "#6bc2ff",
    "fontSize": "14px"
  }
}}%%
flowchart TD
    subgraph Documents Ingesting
        A[/Documents/]
        B[Preprocessing]
        C[Dense vectorize]
        D[Inverted Index Building]
        E[/Dense embeddings/]
        F[/Postings list/]
        G[(Postgres DB)]

        A --> B
        B --> C --> E --> G
        B --> D --> F --> G

        class A source;
        class B,C,D process;
        class E,F output;
        class G storage;
    end

    classDef source fill:#444b6e,stroke:#6bc2ff,color:#fff;
    classDef process fill:#3a506b,stroke:#6bc2ff,color:#fff;
    classDef output fill:#2d6a4f,stroke:#80ed99,color:#fff;
    classDef storage fill:#6a040f,stroke:#ffba08,color:#fff;
```

```mermaid
%%{init: {
  "theme": "dark",
  "themeVariables": {
    "primaryColor": "#2a2f4a",
    "primaryTextColor": "#ffffff",
    "secondaryColor": "#1f2233",
    "tertiaryColor": "#30344d",
    "lineColor": "#b089f0",
    "fontSize": "14px"
  }
}}%%
flowchart TD
    subgraph Users Retrieving
        A[User Queries]
        B[Dense vectorize]
        C[Tokenize]
        D[/Query dense embeddings/]
        E[/Query's tokens/]
        F[Retrieve from DB]
        G[(Postgres DB)]
        H[Fusion]
        I[/Top-K Candidates/]
        J[Rerank with Cross-Encoder]
        K[/Reranked Candidates/]
        L[Prompt Augment]
        M[/Prompt with Context/]
        N[LLM Q&A]
        O[/Final Answers/]

        A --> B --> D --> F
        A --> C --> E --> F
        F --> G --> H --> I --> J --> K --> L
        A --> L --> M --> N --> O

        class A input;
        class B,C,F,H,J,L,N process;
        class D,E,I,K,M,O output;
        class G storage;
    end

    classDef input fill:#444b6e,stroke:#9d4edd,color:#fff;
    classDef process fill:#3a506b,stroke:#b089f0,color:#fff;
    classDef output fill:#2d6a4f,stroke:#80ed99,color:#fff;
    classDef storage fill:#6a040f,stroke:#ffba08,color:#fff;
```

## To-do List

- [ ] Use LangChain instead of custom code + LlamaIndex
- [ ] Use Redis to cache
- [ ] Better separation of concerns (API -> Public Service -> Internal Service -> Repo)
- [ ] Replace local LLM with API-based LLM for text embeddings
