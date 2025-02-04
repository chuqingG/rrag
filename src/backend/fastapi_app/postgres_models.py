from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Define the models
class Base(DeclarativeBase):
    pass


class Item(Base):
    # TODO(chuqing): that's not a good way to set table name
    __tablename__ = "amazon_products"
    asin: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()
    product_url: Mapped[str] = mapped_column()
    stars: Mapped[float] = mapped_column()
    reviews: Mapped[int] = mapped_column()
    price: Mapped[float] = mapped_column()
    list_price: Mapped[float] = mapped_column()
    category_id: Mapped[int] = mapped_column()
    is_best_seller: Mapped[bool] = mapped_column()
    # Embeddings for different models:
    img_embedding: Mapped[Vector] = mapped_column(Vector(1024), nullable=True)  # nomic-embed-text
    img_embedding_bge: Mapped[Vector] = mapped_column(Vector(768), nullable=True)

    def to_dict(self, include_embedding: bool = False):
        model_dict = {column.name: getattr(self, column.name) for column in self.__table__.columns}

        if include_embedding:
            del model_dict['img_embedding']
            model_dict["img_embedding_bge"] = model_dict.get("img_embedding_bge", [])
        else:
            del model_dict['img_embedding_bge']
            del model_dict['img_embedding']
        return model_dict

    def to_str_for_rag(self):
        return f"Title:{self.title} Url:{self.product_url} Price:{self.price} Stars:{self.stars} Review:{self.reviews}"

    def to_str_for_embedding(self):
        return f"Name: {self.name} Description: {self.description} Type: {self.type}"


# Define HNSW index to support vector similarity search
# Use the vector_ip_ops access method (inner product) since these embeddings are normalized

table_name = Item.__tablename__

