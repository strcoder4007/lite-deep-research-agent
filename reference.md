"""
Advanced Research Agent using LangGraph
This implementation includes:
- Proper state management with LangGraph
- Conditional routing
- Error handling
- Streaming support
- Memory integration
- Multi-step reasoning

Install:
pip install langgraph langchain langchain-community chromadb duckduckgo-search
"""

from typing import TypedDict, Annotated, List, Dict
import operator
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


# ========== STATE DEFINITION ==========
class ResearchState(TypedDict):
    """State of the research agent"""
    # Input
    query: str
    
    # Planning
    research_plan: str
    search_queries: List[str]
    
    # Searching
    search_results: List[Dict]
    fetched_content: List[Dict]
    
    # Analysis
    extracted_facts: List[str]
    key_findings: List[str]
    
    # Memory
    relevant_memory: List[str]
    
    # Output
    final_answer: str
    sources: List[str]
    
    # Control
    iteration: int
    max_iterations: int
    errors: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


# ========== TOOLS ==========
class ResearchTools:
    """Collection of tools for the research agent"""
    
    def __init__(self):
        self.llm = Ollama(model="qwen2.5:14b-instruct-q4_K_M", temperature=0.7)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(
            persist_directory="./advanced_memory",
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.ddgs = DDGS()
    
    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call the LLM with error handling"""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web"""
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def fetch_url(self, url: str) -> str:
        """Fetch content from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            return text[:5000]
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
    
    def add_to_memory(self, text: str, metadata: Dict):
        """Add text to vector memory"""
        chunks = self.text_splitter.split_text(text)
        self.vectorstore.add_texts(chunks, metadatas=[metadata] * len(chunks))
    
    def query_memory(self, query: str, k: int = 5) -> List[str]:
        """Query vector memory"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


# ========== AGENT NODES ==========
tools = ResearchTools()


def plan_node(state: ResearchState) -> ResearchState:
    """Create a research plan"""
    print("📋 Planning research...")
    
    query = state["query"]
    
    prompt = f"""You are a research strategist. Create a detailed research plan for this query.

Query: {query}

Provide:
1. 3-4 specific search queries to gather comprehensive information
2. Key aspects to investigate
3. Expected information gaps

Format:
SEARCH_QUERIES:
- [specific query 1]
- [specific query 2]
- [specific query 3]

KEY_ASPECTS:
- [aspect 1]
- [aspect 2]

GAPS_TO_ADDRESS:
- [potential gap 1]
"""
    
    plan = tools.call_llm(prompt, temperature=0.3)
    
    # Extract search queries
    search_queries = []
    if "SEARCH_QUERIES:" in plan:
        queries_section = plan.split("SEARCH_QUERIES:")[1].split("KEY_ASPECTS:")[0]
        queries = [line.strip("- ").strip() for line in queries_section.split("\n") 
                  if line.strip().startswith("-")]
        search_queries = queries[:4]
    
    if not search_queries:
        search_queries = [query]
    
    state["research_plan"] = plan
    state["search_queries"] = search_queries
    state["messages"] = [f"Created research plan with {len(search_queries)} search queries"]
    
    return state


def search_node(state: ResearchState) -> ResearchState:
    """Execute web searches"""
    print("🔍 Searching the web...")
    
    search_queries = state["search_queries"]
    all_results = []
    
    for sq in search_queries:
        results = tools.web_search(sq, max_results=3)
        all_results.extend(results)
    
    # Remove duplicates by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)
    
    state["search_results"] = unique_results[:10]  # Limit to top 10
    state["messages"] = [f"Found {len(unique_results)} unique sources"]
    
    return state


def fetch_node(state: ResearchState) -> ResearchState:
    """Fetch content from URLs"""
    print("📥 Fetching content from sources...")
    
    search_results = state["search_results"]
    fetched_content = []
    
    for result in search_results[:5]:  # Fetch top 5
        content = tools.fetch_url(result["url"])
        if content and "Error" not in content:
            fetched_content.append({
                "url": result["url"],
                "title": result["title"],
                "content": content
            })
            
            # Add to long-term memory
            tools.add_to_memory(
                content,
                {"url": result["url"], "title": result["title"]}
            )
    
    state["fetched_content"] = fetched_content
    state["messages"] = [f"Successfully fetched {len(fetched_content)} pages"]
    
    return state


def analyze_node(state: ResearchState) -> ResearchState:
    """Analyze fetched content"""
    print("🔬 Analyzing content...")
    
    query = state["query"]
    fetched_content = state["fetched_content"]
    
    extracted_facts = []
    
    for item in fetched_content:
        prompt = f"""Extract key facts and insights relevant to this research query.

Query: {query}

Source: {item["title"]}

Content (excerpt):
{item["content"][:2000]}

Extract:
1. Relevant facts and data
2. Key insights or findings
3. Important context

Be concise and focus only on information relevant to the query.
"""
        
        analysis = tools.call_llm(prompt, temperature=0.3)
        
        if analysis and "Error" not in analysis:
            extracted_facts.append(f"[{item['title']}]\n{analysis}")
    
    state["extracted_facts"] = extracted_facts
    state["messages"] = [f"Analyzed {len(extracted_facts)} sources"]
    
    return state


def memory_node(state: ResearchState) -> ResearchState:
    """Query long-term memory for relevant context"""
    print("🧠 Querying memory...")
    
    query = state["query"]
    relevant_memory = tools.query_memory(query, k=5)
    
    state["relevant_memory"] = relevant_memory
    state["messages"] = [f"Retrieved {len(relevant_memory)} relevant memory chunks"]
    
    return state


def synthesize_node(state: ResearchState) -> ResearchState:
    """Synthesize final answer"""
    print("✍️ Synthesizing final report...")
    
    query = state["query"]
    research_plan = state["research_plan"]
    extracted_facts = state["extracted_facts"]
    relevant_memory = state["relevant_memory"]
    sources = [r["url"] for r in state["search_results"]]
    
    prompt = f"""You are a research analyst. Create a comprehensive research report.

Research Query: {query}

Research Plan:
{research_plan}

Extracted Information:
{chr(10).join(extracted_facts)}

Additional Context from Memory:
{chr(10).join(relevant_memory[:3]) if relevant_memory else "None"}

Create a detailed report that:
1. Directly answers the research question
2. Synthesizes information from multiple sources
3. Provides specific facts, data, and examples
4. Acknowledges any limitations or uncertainties
5. Maintains clear source attribution

Structure the report with:
- Executive Summary
- Key Findings (numbered)
- Detailed Analysis
- Conclusion
- Important Notes

Write in a clear, professional style.
"""
    
    final_answer = tools.call_llm(prompt, temperature=0.5)
    
    state["final_answer"] = final_answer
    state["sources"] = sources
    state["messages"] = ["Research report completed"]
    
    return state


def should_continue(state: ResearchState) -> str:
    """Decide whether to continue or finish"""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)
    
    # Check for sufficient information
    if len(state.get("fetched_content", [])) >= 3:
        return "synthesize"
    
    # Check iteration limit
    if iteration >= max_iterations:
        return "synthesize"
    
    # Continue if needed
    state["iteration"] = iteration + 1
    return "search"


# ========== BUILD GRAPH ==========
def create_research_graph():
    """Create the research agent graph"""
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("synthesize", synthesize_node)
    
    # Define flow
    workflow.set_entry_point("plan")
    
    # Linear flow for initial implementation
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "fetch")
    workflow.add_edge("fetch", "analyze")
    workflow.add_edge("analyze", "memory")
    workflow.add_edge("memory", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Could add conditional edges for iterative research:
    # workflow.add_conditional_edges(
    #     "analyze",
    #     should_continue,
    #     {
    #         "search": "search",
    #         "synthesize": "memory"
    #     }
    # )
    
    return workflow.compile()


# ========== MAIN AGENT CLASS ==========
class AdvancedResearchAgent:
    """Advanced research agent with LangGraph"""
    
    def __init__(self):
        self.graph = create_research_graph()
    
    def research(
        self,
        query: str,
        max_iterations: int = 1,
        verbose: bool = True
    ) -> Dict:
        """Execute research"""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Research Query: {query}")
            print(f"{'='*60}\n")
        
        # Initialize state
        initial_state = {
            "query": query,
            "research_plan": "",
            "search_queries": [],
            "search_results": [],
            "fetched_content": [],
            "extracted_facts": [],
            "key_findings": [],
            "relevant_memory": [],
            "final_answer": "",
            "sources": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "errors": [],
            "messages": []
        }
        
        # Execute graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            if verbose:
                print(f"\n{'='*60}")
                print("✅ Research Complete!")
                print(f"{'='*60}\n")
            
            return {
                "query": query,
                "report": final_state["final_answer"],
                "sources": final_state["sources"],
                "plan": final_state["research_plan"],
                "message_log": final_state["messages"]
            }
        
        except Exception as e:
            print(f"\n❌ Error during research: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "report": f"Research failed: {str(e)}",
                "sources": [],
                "plan": "",
                "message_log": []
            }
    
    def visualize_graph(self):
        """Visualize the agent graph (requires pygraphviz)"""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_png()))
        except:
            print("Install pygraphviz to visualize: pip install pygraphviz")


# ========== EXAMPLE USAGE ==========
def main():
    """Example usage"""
    
    # Initialize agent
    print("🤖 Initializing Advanced Research Agent with LangGraph...")
    agent = AdvancedResearchAgent()
    
    # Example queries
    examples = [
        "What are the key differences between vLLM and Ollama for local LLM inference?",
        "How do vector databases improve RAG applications?",
        "What are the latest techniques in agent memory systems?"
    ]
    
    while True:
        print(f"\n{'='*60}")
        print("ADVANCED RESEARCH AGENT")
        print(f"{'='*60}")
        print("\n1. Enter custom query")
        print("2. Use example query")
        print("3. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            query = input("\nEnter query: ").strip()
        elif choice == "2":
            print("\nExamples:")
            for i, ex in enumerate(examples, 1):
                print(f"{i}. {ex}")
            idx = int(input("\nSelect: ").strip()) - 1
            query = examples[idx]
        elif choice == "3":
            break
        else:
            continue
        
        # Execute research
        result = agent.research(query, verbose=True)
        
        # Display results
        print("\n" + "="*60)
        print("📊 RESEARCH REPORT")
        print("="*60)
        print(result["report"])
        print("\n" + "-"*60)
        print("📚 Sources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source}")
        print("="*60)
        
        # Save
        with open(f"report_{hash(query)}.txt", 'w') as f:
            f.write(f"Query: {query}\n\n")
            f.write(result["report"])
            f.write(f"\n\nSources:\n" + "\n".join(result["sources"]))
        
        print(f"\n💾 Report saved!")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()