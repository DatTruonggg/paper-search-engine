from sqlalchemy import Column, String, Text, Integer, DateTime, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import TSVECTOR
from datetime import datetime

Base = declarative_base()


class PaperModel(Base):
    __tablename__ = "papers"
    
    id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    authors = Column(ARRAY(String), nullable=False)
    categories = Column(ARRAY(String), nullable=False)
    year = Column(Integer, nullable=False)
    doi = Column(String, nullable=True)
    url_pdf = Column(String, nullable=True)
    journal_ref = Column(String, nullable=True)
    update_date = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Full-text search vectors
    search_vector = Column(TSVECTOR)
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "year": self.year,
            "doi": self.doi,
            "url_pdf": self.url_pdf,
            "journal_ref": self.journal_ref,
            "update_date": self.update_date
        }
