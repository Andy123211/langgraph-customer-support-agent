"""Vector store for knowledge base using InMemoryVectorStore.

WHAT THIS FILE DOES:
Manages the semantic search functionality for the knowledge base. Converts text documents
into vector embeddings and enables similarity search - meaning you can search by meaning,
not just exact keyword matches.

WHY IT'S IMPORTANT:
Regular keyword search has limitations - it won't find "return policy" if you search for
"refund process". Vector/semantic search understands meaning, so it can find related
content even when the words don't match exactly. This makes the agent much better at
finding relevant information to answer customer questions.
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json
from pathlib import Path
from typing import Callable, Optional
from .retrieval import bm25_relevance, bm25_scores, reciprocal_rank_fusion


class KnowledgeBaseVectorStore:
    """
    Manages the knowledge base vector store.
    
    WHAT IT IS:
    A class that handles converting knowledge base documents into searchable vectors
    and provides semantic search functionality. Think of it like Google's search
    engine but for your company's knowledge base - it finds relevant content based
    on meaning, not just keyword matching.
    
    WHY IT EXISTS:
    Without vector search, you'd need exact keyword matches. With vector search,
    the agent can find relevant information even when customer questions use different
    wording than what's in the knowledge base.
    """

    def __init__(
        self,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker: Optional[Callable[[str, list[Document]], list[Document]]] = None,
    ):
        """
        Initialize the vector store with embeddings model.
        
        WHAT IT DOES:
        Sets up the embeddings model that converts text into vectors (numerical
        representations). These vectors capture the semantic meaning of text.

        WHY IT'S IMPORTANT:
        The embeddings model is what enables semantic search. Different models have
        different strengths - this one (all-MiniLM-L6-v2) is a good balance of
        speed and accuracy.

        Args:
            embeddings_model: HuggingFace model name for embeddings
                            (default is a fast, accurate model good for general use)
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.documents_by_id: dict[str, Document] = {}
        # Optional local or remote cross-encoder adapter. The default path has no
        # additional model or service dependency.
        self.reranker = reranker

    def load_from_json(self, json_path: str) -> None:
        """
        Load knowledge base from JSON file and populate vector store.
        
        WHAT IT DOES:
        Reads the knowledge_base.json file, converts all the content (policies, FAQs,
        product info) into Document objects, then converts those into vector embeddings
        and stores them in the vector store.

        WHY IT'S IMPORTANT:
        This is the setup step that makes the knowledge base searchable. Without this,
        the search tools wouldn't have anything to search. This converts all your
        company knowledge into a format that can be semantically searched.

        Args:
            json_path: Path to knowledge_base.json file
        """
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Convert JSON to documents
        documents = self._json_to_documents(data)

        # Create vector store
        self.documents_by_id = {}
        for index, doc in enumerate(documents):
            document_id = f"kb-{index}"
            self.documents_by_id[document_id] = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "source_id": document_id},
            )
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=list(self.documents_by_id.values()),
            embedding=self.embeddings,
        )

        print(f"✅ Loaded {len(documents)} documents into vector store")

    def _json_to_documents(self, data: dict) -> list[Document]:
        """
        Convert JSON knowledge base to LangChain documents.
        
        WHAT IT DOES:
        Takes the raw JSON data and converts each piece of information (policies,
        FAQs, product info) into LangChain Document objects. Each document includes
        the content (text) and metadata (category, type) for filtering.

        WHY IT'S IMPORTANT:
        LangChain's vector store works with Document objects, not raw JSON. This method
        structures the knowledge base into a format that can be vectorized and searched.
        The metadata (category, type) allows filtering searches to specific areas.

        Args:
            data: Knowledge base dictionary from JSON file

        Returns:
            List of Document objects ready for vectorization
        """
        documents = []
        kb = data.get("knowledge_base", {})

        # Process policies
        policies = kb.get("policies", {})

        # Returns policy
        if "returns" in policies:
            returns = policies["returns"]
            content = f"""Return Policy:
- Time limit: {returns.get('time_limit', 'N/A')}
- Condition: {returns.get('condition', 'N/A')}
- Refund processing time: {returns.get('refund_time', 'N/A')}
- Return shipping cost: Free for defective items, ${returns.get('return_shipping', {}).get('other', 'N/A')} for other reasons
"""
            documents.append(Document(
                page_content=content,
                metadata={"category": "return", "type": "policy"}
            ))

        # Shipping policy
        if "shipping" in policies:
            shipping = policies["shipping"]
            content = "Shipping Options:\n"
            for method, details in shipping.items():
                content += f"- {method.title()}: {details.get('time', 'N/A')} - {details.get('cost', 'N/A')}\n"
            documents.append(Document(
                page_content=content,
                metadata={"category": "shipping", "type": "policy"}
            ))

        # Warranty policy
        if "warranty" in policies:
            warranty = policies["warranty"]
            content = f"""Warranty Information:
- Standard warranty: {warranty.get('standard', 'N/A')}
- Coverage: {warranty.get('coverage', 'N/A')}
- Extended warranty available: {'Yes' if warranty.get('extended_available') else 'No'}
"""
            documents.append(Document(
                page_content=content,
                metadata={"category": "product", "type": "policy"}
            ))

        # Payment policy
        if "payment" in policies:
            payment = policies["payment"]
            methods = ", ".join(payment.get('methods', []))
            content = f"""Payment Information:
- Accepted methods: {methods}
- Payment processor: {payment.get('processor', 'N/A')}
- Security: {payment.get('security', 'N/A')}
- Payment timing: Charged when order ships
"""
            documents.append(Document(
                page_content=content,
                metadata={"category": "payment", "type": "policy"}
            ))

        # Products
        products = kb.get("products", {})
        if "categories" in products:
            categories = ", ".join(products["categories"])
            content = f"Product Categories: {categories}"
            documents.append(Document(
                page_content=content,
                metadata={"category": "product", "type": "info"}
            ))

        if "top_products" in products:
            for product in products["top_products"]:
                features = ", ".join(product.get("features", []))
                content = f"""{product.get('name', 'Unknown Product')}
Price: {product.get('price', 'N/A')}
Features: {features}
"""
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "category": "product",
                        "type": "product_info",
                        "product_name": product.get("name", "")
                    }
                ))

        # FAQs
        for faq in kb.get("faq", []):
            content = f"""Question: {faq.get('question', '')}
Answer: {faq.get('answer', '')}
"""
            documents.append(Document(
                page_content=content,
                metadata={"category": "general", "type": "faq"}
            ))

        # Add general support info
        documents.append(Document(
            page_content="""Customer Support Information:
- Hours: Monday-Friday, 9 AM - 6 PM EST
- Contact methods: Chat, email (support@store.com), phone (1-800-SUPPORT)
- Order modifications: Available within 1 hour of placing order
- Price matching: Available on identical items from authorized retailers
- Gift wrapping: Available for $5 per item with custom messages
""",
            metadata={"category": "general", "type": "info"}
        ))

        return documents

    def search(self, query: str, k: int = 3, filter_category: Optional[str] = None) -> str:
        """Return concise hybrid-retrieval results for a customer query."""
        categories = [filter_category] if filter_category else None
        results = self.search_with_scores(query, k=k, filter_categories=categories)
        if not results:
            return "No relevant information found in the knowledge base."
        return "\n\n".join(doc.page_content.strip() for doc, _ in results)

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_categories: Optional[list[str]] = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """Retrieve with BM25 and vector search, fuse ranks with RRF, and rerank optionally.

        Returned values are heuristic relevance indicators, not calibrated confidence
        probabilities. A supplied reranker receives the query and fused candidate list
        and returns those documents in its preferred order.
        """
        if self.vector_store is None or not self.documents_by_id:
            return []

        eligible = {
            document_id: document
            for document_id, document in self.documents_by_id.items()
            if not filter_categories
            or document.metadata.get("category") in filter_categories
        }
        if not eligible:
            return []

        def metadata_filter(document: Document) -> bool:
            return not filter_categories or document.metadata.get("category") in filter_categories

        candidate_k = min(len(eligible), max(k * 2, 10))
        lexical_raw = bm25_scores(
            query,
            {document_id: document.page_content for document_id, document in eligible.items()},
        )
        lexical_ranked = sorted(
            (document_id for document_id, score in lexical_raw.items() if score > 0),
            key=lambda document_id: (-lexical_raw[document_id], document_id),
        )[:candidate_k]

        # InMemoryVectorStore returns cosine similarity here (higher is better).
        vector_hits = self.vector_store.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=metadata_filter,
        )
        vector_ranked: list[str] = []
        vector_relevance: dict[str, float] = {}
        for document, score in vector_hits:
            document_id = document.metadata.get("source_id")
            if document_id not in eligible:
                # Compatibility fallback for vector-store adapters that omit ids.
                document_id = next(
                    (key for key, value in eligible.items() if value.page_content == document.page_content),
                    None,
                )
            if document_id is None:
                continue
            vector_ranked.append(document_id)
            vector_relevance[document_id] = max(0.0, min(1.0, float(score)))

        fused_ids = reciprocal_rank_fusion(lexical_ranked, vector_ranked)
        if self.reranker is not None and fused_ids:
            candidates = [eligible[document_id] for document_id in fused_ids]
            reranked = self.reranker(query, candidates)
            reranked_ids = [
                document.metadata.get("source_id") for document in reranked
                if document.metadata.get("source_id") in eligible
            ]
            fused_ids = reranked_ids + [
                document_id for document_id in fused_ids if document_id not in reranked_ids
            ]

        ranked_results = []
        for document_id in fused_ids:
            document = eligible[document_id]
            relevance = max(
                vector_relevance.get(document_id, 0.0),
                bm25_relevance(lexical_raw.get(document_id, 0.0)),
            )
            if relevance >= score_threshold:
                ranked_results.append((document, relevance))
            if len(ranked_results) >= k:
                break
        return ranked_results


# Global instance - initialized lazily
# WHAT: A module-level variable that holds the vector store instance
# WHY: We want a single instance that loads the knowledge base once, then gets reused.
#      This is a "singleton" pattern - ensures all tools use the same vector store.
_vector_store_instance: Optional[KnowledgeBaseVectorStore] = None


def get_vector_store() -> KnowledgeBaseVectorStore:
    """
    Get or create the global vector store instance.
    
    WHAT IT DOES:
    Returns the global vector store instance, creating and loading it if it doesn't exist yet.
    This is lazy initialization - the knowledge base only loads when first needed.
    
    WHY IT'S IMPORTANT:
    Provides a convenient way for tools to access the vector store without worrying about
    initialization. Tools just call get_vector_store() and get a ready-to-use instance.
    The singleton pattern ensures we only load the knowledge base once, which is more efficient.
    
    Returns:
        The global KnowledgeBaseVectorStore instance (loaded and ready to search)
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        _vector_store_instance = KnowledgeBaseVectorStore()

        # Load knowledge base from default location
        kb_path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.json"
        if kb_path.exists():
            _vector_store_instance.load_from_json(str(kb_path))
        else:
            print(f"⚠️ Warning: Knowledge base not found at {kb_path}")

    return _vector_store_instance
