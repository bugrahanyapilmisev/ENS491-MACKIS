from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Float, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func, text
from sqlalchemy.dialects.postgresql import UUID, JSONB, BYTEA
from pgvector.sqlalchemy import Vector  # pgvector desteği
from database import Base

# =====================================================================
# 1. USERS & AUTH (KULLANICILAR & YETKİLENDİRME)
# =====================================================================

class User(Base):
    __tablename__ = "users"

    # gen_random_uuid() postgres tarafında pgcrypto eklentisi gerektirir
    user_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    email = Column(String, unique=True)
    display_name = Column(String)
    # role ve status için CheckConstraint veritabanı seviyesinde kontrol sağlar
    role = Column(String, nullable=False, server_default='student')
    status = Column(String, nullable=False, server_default='active')
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    query_sessions = relationship("QuerySession", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")

    __table_args__ = (
        CheckConstraint("role IN ('undergrad','grad','alumni','staff','admin','guest')", name='check_role'),
        CheckConstraint("status IN ('active','disabled')", name='check_status'),
    )

class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    title = Column(String)
    archived = Column(Boolean, nullable=False, server_default=text("false"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    user = relationship("User", back_populates="conversations")
    query_sessions = relationship("QuerySession", back_populates="conversation")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class QuerySession(Base):
    __tablename__ = "query_sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"))
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id", ondelete="SET NULL"))
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    lang = Column(String, nullable=False, server_default='tr')
    client_meta = Column(JSONB)

    # İlişkiler
    user = relationship("User", back_populates="query_sessions")
    conversation = relationship("Conversation", back_populates="query_sessions")
    events = relationship("QueryEvent", back_populates="session", cascade="all, delete-orphan")

class QueryEvent(Base):
    __tablename__ = "query_events"

    query_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("query_sessions.session_id", ondelete="CASCADE"), nullable=False)
    query_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    session = relationship("QuerySession", back_populates="events")
    answer = relationship("Answer", uselist=False, back_populates="query_event", cascade="all, delete-orphan")
    retrieval_hits = relationship("RetrievalHit", back_populates="query_event", cascade="all, delete-orphan")
    # ChatMessage ile ilişki (Forward Ref)
    chat_messages = relationship("ChatMessage", back_populates="query_event")

class Answer(Base):
    __tablename__ = "answers"

    answer_id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_events.query_id", ondelete="CASCADE"), nullable=False, unique=True)
    text = Column(Text, nullable=False)
    model_name = Column(String)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    query_event = relationship("QueryEvent", back_populates="answer")
    citations = relationship("AnswerCitation", back_populates="answer", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="answer")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    
    # Traceability Linkleri
    query_id = Column(Integer, ForeignKey("query_events.query_id", ondelete="SET NULL"))
    answer_id = Column(Integer, ForeignKey("answers.answer_id", ondelete="SET NULL"))
    
    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")
    query_event = relationship("QueryEvent", back_populates="chat_messages")
    answer = relationship("Answer", back_populates="chat_messages")

    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','system')", name='check_role_msg'),
    )

# =====================================================================
# 2. CONTENT & CHUNKING (İÇERİK & PARÇALAMA)
# =====================================================================

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String, nullable=False)
    source_uri = Column(String)
    title = Column(String)
    lang = Column(String, nullable=False, server_default='tr')
    department = Column(String)
    hash = Column(String, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    kg_nodes = relationship("KGNode", back_populates="document")
    ingest_artifacts = relationship("IngestArtifact", back_populates="document")

    __table_args__ = (
        CheckConstraint("source_type IN ('url','pdf','html','md','email','other')", name='check_source_type'),
    )

class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    ordinal = Column(Integer, nullable=False)
    section = Column(String)
    page_num = Column(Integer)
    content = Column(Text, nullable=False)
    content_html = Column(Text)
    tokens = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    document = relationship("Document", back_populates="chunks")
    embedding_data = relationship("ChunkEmbedding", uselist=False, back_populates="chunk", cascade="all, delete-orphan")
    retrieval_hits = relationship("RetrievalHit", back_populates="chunk")
    citations = relationship("AnswerCitation", back_populates="chunk")

class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id = Column(Integer, ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True)
    # SQL'de vector(1024) istenmiş.
    embedding = Column(Vector(1024), nullable=False)
    model = Column(String)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # İlişkiler
    chunk = relationship("Chunk", back_populates="embedding_data")

# =====================================================================
# 3. KNOWLEDGE GRAPH (BİLGİ GRAFİĞİ)
# =====================================================================

class KGNode(Base):
    __tablename__ = "kg_nodes"

    node_id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String, nullable=False)
    name = Column(String, nullable=False)
    doc_id = Column(Integer, ForeignKey("documents.document_id", ondelete="SET NULL"))
    data = Column(JSONB)

    # İlişkiler
    document = relationship("Document", back_populates="kg_nodes")
    # Graph ilişkileri (Source ve Target olarak)
    edges_out = relationship("KGEdge", foreign_keys="[KGEdge.src]", back_populates="source_node", cascade="all, delete-orphan")
    edges_in = relationship("KGEdge", foreign_keys="[KGEdge.dst]", back_populates="target_node", cascade="all, delete-orphan")

class KGEdge(Base):
    __tablename__ = "kg_edges"

    edge_id = Column(Integer, primary_key=True, autoincrement=True)
    src = Column(Integer, ForeignKey("kg_nodes.node_id", ondelete="CASCADE"), nullable=False)
    dst = Column(Integer, ForeignKey("kg_nodes.node_id", ondelete="CASCADE"), nullable=False)
    type = Column(String, nullable=False)
    weight = Column(Float, nullable=False, server_default=text("1.0"))

    # İlişkiler
    source_node = relationship("KGNode", foreign_keys=[src], back_populates="edges_out")
    target_node = relationship("KGNode", foreign_keys=[dst], back_populates="edges_in")

    __table_args__ = (
        UniqueConstraint('src', 'dst', 'type', name='uq_kgedges'),
    )

# =====================================================================
# 4. RETRIEVAL & ANSWERS (ERİŞİM & CEVAPLAR - Detaylar)
# =====================================================================

class RetrievalHit(Base):
    __tablename__ = "retrieval_hits"

    hit_id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_events.query_id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("chunks.chunk_id", ondelete="CASCADE"), nullable=False)
    rank = Column(Integer, nullable=False)
    score_dense = Column(Float)
    score_lex = Column(Float)
    score_graph = Column(Float)

    # İlişkiler
    query_event = relationship("QueryEvent", back_populates="retrieval_hits")
    chunk = relationship("Chunk", back_populates="retrieval_hits")

    __table_args__ = (
        UniqueConstraint('query_id', 'chunk_id', name='uq_rhits_query_chunk'),
        UniqueConstraint('query_id', 'rank', name='uq_rhits_query_rank'),
    )

class AnswerCitation(Base):
    __tablename__ = "answer_citations"

    answer_id = Column(Integer, ForeignKey("answers.answer_id", ondelete="CASCADE"), primary_key=True)
    order_idx = Column(Integer, primary_key=True) # Composite PK
    chunk_id = Column(Integer, ForeignKey("chunks.chunk_id", ondelete="CASCADE"), nullable=True)

    # İlişkiler
    answer = relationship("Answer", back_populates="citations")
    chunk = relationship("Chunk", back_populates="citations")

    __table_args__ = (
        UniqueConstraint('answer_id', 'chunk_id', name='uq_cite_once'),
    )

# =====================================================================
# 5. INGESTION OPS (YÜKLEME İŞLEMLERİ)
# =====================================================================

class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String, nullable=False)
    source_uri = Column(String)
    status = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at = Column(DateTime(timezone=True))
    error_msg = Column(Text)

    # İlişkiler
    artifacts = relationship("IngestArtifact", back_populates="job", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("source_type IN ('url','pdf','html','md','email','other')", name='check_ingest_source'),
        CheckConstraint("status IN ('queued','running','succeeded','failed')", name='check_ingest_status'),
    )

class IngestArtifact(Base):
    __tablename__ = "ingest_artifacts"

    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("ingest_jobs.job_id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    stage = Column(String, nullable=False)
    notes = Column(Text)

    # İlişkiler
    job = relationship("IngestJob", back_populates="artifacts")
    document = relationship("Document", back_populates="ingest_artifacts")

    __table_args__ = (
        UniqueConstraint('job_id', 'document_id', 'stage', name='uq_artifact_uniqueness'),
        CheckConstraint("stage IN ('fetched','parsed','chunked','embedded','graphified')", name='check_artifact_stage'),
    )