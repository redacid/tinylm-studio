"""SQL queries over the data, for the SQL tab: the tables with their columns, and running one."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from engine.api.deps import require_api_key
from engine.rag.query import DEFAULT_LIMIT, MAX_LIMIT, DataQuery, QueryError

router = APIRouter(prefix="/api/query", dependencies=[Depends(require_api_key)], tags=["knowledge"])


class QueryRequest(BaseModel):
    sql: str
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    # A DELETE only counts its rows unless confirmed.
    confirm: bool = False


def data_query(request: Request) -> DataQuery:
    return request.app.state.data_query


@router.get("/schema")
def schema(query: DataQuery = Depends(data_query)) -> dict:
    """The tables a query may read, their columns with types, and example queries."""
    try:
        return query.schema()
    except QueryError as e:
        raise HTTPException(400, str(e)) from None


@router.post("")
def run(body: QueryRequest, query: DataQuery = Depends(data_query)) -> dict:
    """Runs one SELECT over the data (its columns and the first rows), or a DELETE from a data
    table: it counts the rows, and deletes them when confirmed."""
    try:
        return query.run(body.sql, body.limit, body.confirm)
    except QueryError as e:
        raise HTTPException(400, str(e)) from None
