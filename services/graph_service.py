import json
import os
from typing import List, Dict, Any
import requests
from sqlalchemy.orm import Session
from sqlalchemy import text
import models
from database import SessionLocal

# .env'den ayarları çekelim
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b") # Veya llama3.2

class GraphService:
    def __init__(self, db: Session):
        self.db = db

    def extract_triples_from_text(self, chunk_text: str) -> List[Dict[str, Any]]:
        """
        LLM kullanarak metinden [Subject, Relation, Object] üçlülerini çıkarır.
        Örn: "Ahmet Hoca CS406 dersini veriyor" -> 
        {"head": "Ahmet Hoca", "type": "gives_lecture", "tail": "CS406"}
        """
        
        system_prompt = (
            "Sen bir Bilgi Grafiği (Knowledge Graph) uzmanısın. "
            "Görevin verilen metindeki varlıkları (Entities) ve aralarındaki ilişkileri (Relationships) çıkarmaktır.\n"
            "Çıktı FORMATI kesinlikle JSON olmalıdır:\n"
            "{\n"
            '  "triples": [\n'
            '    {"head": "Varlık1", "type": "ilişki_türü", "tail": "Varlık2", "head_type": "Person/Course/Date/...", "tail_type": "..."}\n'
            '  ]\n'
            "}\n"
            "Kurallar:\n"
            "1. Varlık isimlerini mümkün olduğunca standartlaştır (örn: 'Sabancı Üniv.' -> 'Sabancı Üniversitesi').\n"
            "2. İlişki türlerini snake_case yap (örn: 'dersi_verir', 'bağlıdır', 'ön_şartıdır').\n"
            "3. Sadece metinde KESİN olan bilgileri çıkar.\n"
        )

        user_prompt = f"Metin:\n{chunk_text}\n\nJSON Çıktısı:"

        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "format": "json", # Ollama JSON modu
                    "options": {"temperature": 0.0} 
                },
                timeout=300
            )
            data = resp.json()
            content = data["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("triples", [])
        except Exception as e:
            print(f"❌ Graph Extraction Hatası: {e}")
            return []

    def save_triples_to_db(self, triples: List[Dict], doc_id: int = None):
        """
        Çıkarılan üçlüleri veritabanına kaydeder.
        Hata toleranslı (Fault-Tolerant) versiyon.
        """
        # 1. Tekrarları Python tarafında temizle
        seen_triples = set()
        unique_triples = []

        for t in triples:
            head = t.get("head", "").strip()
            tail = t.get("tail", "").strip()
            rel_type = t.get("type", "").strip()
            
            if not head or not tail: continue

            key = (head.lower(), rel_type.lower(), tail.lower())
            if key not in seen_triples:
                seen_triples.add(key)
                unique_triples.append(t)

        # 2. Veritabanına Ekleme Döngüsü
        for triple in unique_triples:
            try:
                # Nested transaction başlatıyoruz (Hata olursa sadece bu işlem geri alınsın)
                with self.db.begin_nested():
                    head_node = self._get_or_create_node(triple["head"], triple.get("head_type", "Entity"), doc_id)
                    tail_node = self._get_or_create_node(triple["tail"], triple.get("tail_type", "Entity"), doc_id)
                    
                    self._create_edge_if_not_exists(head_node.node_id, tail_node.node_id, triple["type"])
            
            except Exception as e:
                # Bu spesifik üçlüde hata olursa (Duplicate vb.) atla ve devam et
                print(f"⚠️ İlişki eklenirken hata (Atlandı): {triple['head']} -> {triple['tail']} | Hata: {e}")
                continue

        # 3. Her şey bitince kalıcı yap
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"❌ Toplu Commit Hatası: {e}")

    def _get_or_create_node(self, name, label, doc_id):
        node = self.db.query(models.KGNode).filter(models.KGNode.name == name).first()
        if not node:
            node = models.KGNode(name=name, label=label, doc_id=doc_id)
            self.db.add(node)
            self.db.flush() # ID almak için şart
        return node

    def _create_edge_if_not_exists(self, src_id, dst_id, rel_type):
        # Sadece varlık kontrolü
        exists = self.db.query(models.KGEdge).filter(
            models.KGEdge.src == src_id,
            models.KGEdge.dst == dst_id,
            models.KGEdge.type == rel_type
        ).first()
        
        if not exists:
            edge = models.KGEdge(src=src_id, dst=dst_id, type=rel_type)
            self.db.add(edge)
            self.db.flush()

    def search_graph(self, query_entity: str, depth: int = 1) -> List[str]:
        """
        Bir varlık ismi verildiğinde (örn: "CS406"), grafikte ona bağlı olan bilgileri getirir.
        """
        # Basit bir SQL sorgusu ile komşuları bulalım
        sql = text("""
            SELECT n1.name as source, e.type, n2.name as target
            FROM kg_edges e
            JOIN kg_nodes n1 ON e.src = n1.node_id
            JOIN kg_nodes n2 ON e.dst = n2.node_id
            WHERE n1.name ILIKE :entity OR n2.name ILIKE :entity
            LIMIT 20
        """)
        
        results = self.db.execute(sql, {"entity": f"%{query_entity}%"}).fetchall()
        
        knowledge = []
        for r in results:
            knowledge.append(f"{r[0]} --[{r[1]}]--> {r[2]}")
            
        return knowledge