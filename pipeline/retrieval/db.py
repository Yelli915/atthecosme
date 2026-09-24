"""성분 마스터 DB (SQLite).

`data/ingredient_db/seed_ingredients.csv`는 코드 동작 확인용 샘플 15건이다.
실제 마스터 DB의 출처·규모는 아직 미확정 — records/01-사실정리.md 에서 확인 후
이 CSV를 실제 데이터로 교체하거나 별도 적재 스크립트를 추가한다.
"""

import csv
import sqlite3
from dataclasses import dataclass

from pipeline.config import INGREDIENT_DB_PATH, INGREDIENT_SEED_CSV

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingredients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inci_name TEXT NOT NULL,
    kr_name TEXT,
    aliases TEXT  -- '|'로 구분된 이표기 목록
);
"""


@dataclass
class Ingredient:
    id: int
    inci_name: str
    kr_name: str | None
    aliases: list[str]

    def search_text(self) -> str:
        """검색 인덱싱에 쓸 표면형 전체(INCI명 + 한글명 + 이표기)를 합친 문자열."""
        parts = [self.inci_name]
        if self.kr_name:
            parts.append(self.kr_name)
        parts.extend(self.aliases)
        return " ".join(parts)


def build_db(csv_path=INGREDIENT_SEED_CSV, db_path=INGREDIENT_DB_PATH) -> None:
    """CSV로부터 SQLite DB를 (재)생성한다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS ingredients")
        conn.execute(SCHEMA)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                (
                    row["inci_name"].strip(),
                    row["kr_name"].strip() or None,
                    row["aliases"].strip() or None,
                )
                for row in reader
            ]
        conn.executemany(
            "INSERT INTO ingredients (inci_name, kr_name, aliases) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_all(db_path=INGREDIENT_DB_PATH) -> list[Ingredient]:
    """DB의 전체 성분을 로드한다. BM25/Dense 인덱스 구축의 입력."""
    if not db_path.exists():
        build_db(db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT id, inci_name, kr_name, aliases FROM ingredients")
        result = []
        for id_, inci_name, kr_name, aliases in cur.fetchall():
            alias_list = aliases.split("|") if aliases else []
            result.append(Ingredient(id=id_, inci_name=inci_name, kr_name=kr_name, aliases=alias_list))
        return result
    finally:
        conn.close()


def get_by_id(ingredient_id: int, db_path=INGREDIENT_DB_PATH) -> Ingredient | None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT id, inci_name, kr_name, aliases FROM ingredients WHERE id = ?",
            (ingredient_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        id_, inci_name, kr_name, aliases = row
        alias_list = aliases.split("|") if aliases else []
        return Ingredient(id=id_, inci_name=inci_name, kr_name=kr_name, aliases=alias_list)
    finally:
        conn.close()


if __name__ == "__main__":
    build_db()
    print(f"성분 {len(load_all())}건 적재 완료 → {INGREDIENT_DB_PATH}")
