#!/usr/bin/env python3
"""Import archived Moltbook Observatory data from Hugging Face into PostgreSQL.

The public archive exposes posts, agents, comments, and submolts as separate HF
dataset subsets. This importer samples a requested number of posts, pulls in the
related dependency closure, and writes rows into the current PostgreSQL schema
with idempotent inserts.

Votes are supported only if the dataset exposes a dedicated votes subset in the
future. The current archive card does not publish one, so vote import becomes a
no-op unless a compatible subset is available.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg
from datasets import load_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "liangyucheng/moltbook-observatory-archive"
DEFAULT_SPLIT = "archive"
DEFAULT_POSTS_DATA_FILES = "data/posts/2026-01-2*.parquet"
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_SEED = 42

AGENT_NAMESPACE = uuid.UUID("11111111-1111-1111-1111-111111111111")
SUBMOLT_NAMESPACE = uuid.UUID("22222222-2222-2222-2222-222222222222")
POST_NAMESPACE = uuid.UUID("33333333-3333-3333-3333-333333333333")
COMMENT_NAMESPACE = uuid.UUID("44444444-4444-4444-4444-444444444444")
VOTE_NAMESPACE = uuid.UUID("55555555-5555-5555-5555-555555555555")


logger = logging.getLogger("import_hf_archive")


def configure_external_logging() -> None:
    for name in (
        "datasets",
        "datasets.builder",
        "datasets.iterable_dataset",
        "huggingface_hub",
        "httpx",
        "urllib3",
        "filelock",
        "fsspec",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def count_with_progress(rows: Iterable[dict[str, Any]], label: str, log_every: int = 100000) -> int:
    count = 0
    for count, _ in enumerate(rows, start=1):
        if count % log_every == 0:
            logger.info("%s scanned %s row(s)", label, count)
    return count


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_local_env() -> None:
    for candidate in (ROOT_DIR / ".env", ROOT_DIR.parent / ".env"):
        load_dotenv_file(candidate)


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_optional_text(value: Any, max_length: int | None = None) -> str | None:
    text = normalize_text(value, "")
    if text and max_length:
        text = text[:max_length]
    return text or None


def truncate_text(value: str, max_length: int) -> str:
    """Truncate text to specified max length for database fields."""
    if not value:
        return value
    return value[:max_length]


def normalize_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def pick_value(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def stable_uuid(namespace: uuid.UUID, source_id: Any) -> str:
    return str(uuid.uuid5(namespace, normalize_text(source_id, "unknown")))


def placeholder_hash(prefix: str, source_id: Any) -> str:
    payload = f"{prefix}:{normalize_text(source_id, 'unknown')}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_stream(dataset: str, split: str, table: str):
    logger.info("Opening dataset subset: %s/%s [%s]", dataset, split, table)
    stream = load_dataset(dataset, table, split=split, streaming=True)
    logger.info("Dataset subset opened successfully: %s", table)
    return stream


def load_posts_dataset(dataset: str, data_files: str):
    logger.info("Loading posts dataset with data files: %s", data_files)
    ds = load_dataset(
        dataset,
        "posts",
        data_files=data_files,
        split="train",
    )
    logger.info("Posts dataset loaded successfully")
    return ds


def reservoir_sample(rows: Iterable[dict[str, Any]], sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if len(sample) < sample_size:
            sample.append(row)
            continue

        swap_index = rng.randint(0, index - 1)
        if swap_index < sample_size:
            sample[swap_index] = row

    return sample


def get_source_agent_key(row: dict[str, Any]) -> tuple[str | None, str]:
    agent_id = normalize_optional_text(pick_value(row, "agent_id", "author_id", "author", "agent"))
    agent_name = normalize_optional_text(pick_value(row, "agent_name", "author_name", "name"))

    if agent_id:
        return agent_id, agent_name or agent_id
    if agent_name:
        return agent_name, agent_name
    return None, ""


def get_source_submolt_name(row: dict[str, Any]) -> str | None:
    submolt = pick_value(row, "submolt", "submolt_name", "community")
    if isinstance(submolt, dict):
        submolt = submolt.get("name")
    return normalize_optional_text(submolt)


def get_source_post_id(row: dict[str, Any]) -> str | None:
    return normalize_optional_text(pick_value(row, "id", "post_id"))


def get_source_comment_id(row: dict[str, Any]) -> str | None:
    return normalize_optional_text(pick_value(row, "id", "comment_id"))


def get_source_parent_id(row: dict[str, Any]) -> str | None:
    return normalize_optional_text(pick_value(row, "parent_id", "reply_to", "parent"))


def get_target_id(row: dict[str, Any]) -> str | None:
    return normalize_optional_text(pick_value(row, "target_id", "target"))


def timestamp_value(row: dict[str, Any], *keys: str) -> Any:
    value = pick_value(row, *keys)
    return normalize_optional_text(value)


@dataclass
class ImportStats:
    sampled_posts: int = 0
    inserted_agents: int = 0
    skipped_agents: int = 0
    inserted_submolts: int = 0
    skipped_submolts: int = 0
    inserted_posts: int = 0
    skipped_posts: int = 0
    inserted_comments: int = 0
    skipped_comments: int = 0
    inserted_votes: int = 0
    skipped_votes: int = 0


class ArchiveImporter:
    def __init__(self, database_url: str, dataset: str, split: str, seed: int, batch_size: int, posts_data_files: str) -> None:
        self.database_url = database_url
        self.dataset = dataset
        self.split = split
        self.seed = seed
        self.batch_size = batch_size
        self.posts_data_files = posts_data_files
        self.rng = random.Random(seed)
        self.stats = ImportStats()

        self.agent_rows_by_source_id: dict[str, dict[str, Any]] = {}
        self.submolt_rows_by_name: dict[str, dict[str, Any]] = {}
        self.post_rows_by_source_id: dict[str, dict[str, Any]] = {}
        self.comment_rows_by_source_id: dict[str, dict[str, Any]] = {}
        self.vote_rows_by_source_id: dict[str, dict[str, Any]] = {}

        self.agent_id_map: dict[str, str] = {}
        self.submolt_id_map: dict[str, str] = {}
        self.post_id_map: dict[str, str] = {}
        self.comment_id_map: dict[str, str] = {}
        self.vote_id_map: dict[str, str] = {}
        self.submolt_creator_sources: dict[str, str] = {}

    def sample_posts(self, count: int) -> list[dict[str, Any]]:
        logger.info("Sampling %s post(s) from %s/%s", count, self.dataset, self.split)
        try:
            posts_dataset = load_posts_dataset(self.dataset, self.posts_data_files)
        except Exception as error:  # pragma: no cover - external dataset availability
            raise RuntimeError(f"Failed to load posts subset: {error}") from error

        if len(posts_dataset) <= count:
            sampled = [dict(row) for row in posts_dataset]
        else:
            sampled = [dict(row) for row in posts_dataset.shuffle(seed=self.seed).select(range(count))]

        self.stats.sampled_posts = len(sampled)
        logger.info("Sampled %s post(s) successfully", self.stats.sampled_posts)
        return sampled

    def collect_related_rows(self, sampled_posts: list[dict[str, Any]]) -> None:
        logger.info("Collecting related rows for sampled posts")
        selected_post_ids: set[str] = set()
        selected_submolts: set[str] = set()
        selected_agent_ids: set[str] = set()
        selected_comment_ids: set[str] = set()

        for post in sampled_posts:
            post_id = get_source_post_id(post)
            if not post_id:
                continue

            selected_post_ids.add(post_id)
            self.post_rows_by_source_id[post_id] = post

            agent_source_id, _ = get_source_agent_key(post)
            if agent_source_id:
                selected_agent_ids.add(agent_source_id)

            submolt_name = get_source_submolt_name(post)
            if submolt_name:
                selected_submolts.add(submolt_name)
                if submolt_name not in self.submolt_creator_sources and agent_source_id:
                    self.submolt_creator_sources[submolt_name] = agent_source_id

        try:
            comments_stream = load_stream(self.dataset, self.split, "comments")
            logger.info("Scanning comments for related rows")
            for index, row in enumerate(comments_stream, start=1):
                if index % 100000 == 0:
                    logger.info("Comments scan progress: %s row(s) processed", index)
                post_id = normalize_optional_text(pick_value(row, "post_id"))
                if post_id not in selected_post_ids:
                    continue

                comment_id = get_source_comment_id(row)
                if not comment_id:
                    continue

                self.comment_rows_by_source_id[comment_id] = row
                selected_comment_ids.add(comment_id)

                agent_source_id, _ = get_source_agent_key(row)
                if agent_source_id:
                    selected_agent_ids.add(agent_source_id)
        except Exception as error:
            raise RuntimeError(f"Failed to load comments subset: {error}") from error

        try:
            agents_stream = load_stream(self.dataset, self.split, "agents")
            for row in agents_stream:
                source_id, source_name = get_source_agent_key(row)
                if not source_id:
                    continue

                if source_id in selected_agent_ids or source_name in selected_agent_ids:
                    self.agent_rows_by_source_id[source_id] = row
        except Exception as error:
            raise RuntimeError(f"Failed to load agents subset: {error}") from error

        try:
            submolts_stream = load_stream(self.dataset, self.split, "submolts")
            for row in submolts_stream:
                name = normalize_optional_text(pick_value(row, "name"))
                if not name or name not in selected_submolts:
                    continue
                self.submolt_rows_by_name[name] = row
        except Exception as error:
            raise RuntimeError(f"Failed to load submolts subset: {error}") from error

        try:
            votes_stream = load_stream(self.dataset, self.split, "votes")
            for row in votes_stream:
                vote_id = normalize_optional_text(pick_value(row, "id", "vote_id"))
                if not vote_id:
                    continue

                target_id = get_target_id(row)
                if target_id not in selected_post_ids and target_id not in selected_comment_ids:
                    continue

                self.vote_rows_by_source_id[vote_id] = row

                agent_source_id, _ = get_source_agent_key(row)
                if agent_source_id:
                    selected_agent_ids.add(agent_source_id)
        except Exception:
            self.vote_rows_by_source_id = {}

        for source_id in selected_agent_ids:
            self.agent_rows_by_source_id.setdefault(
                source_id,
                {
                    "id": source_id,
                    "name": source_id,
                    "description": "",
                    "karma": 0,
                    "follower_count": 0,
                    "following_count": 0,
                    "is_claimed": False,
                    "created_at": None,
                    "last_seen_at": None,
                },
            )

        for name in selected_submolts:
            self.submolt_rows_by_name.setdefault(
                name,
                {
                    "name": name,
                    "display_name": name,
                    "description": "",
                    "subscriber_count": 0,
                    "post_count": 0,
                    "created_at": None,
                },
            )

        logger.info(
            "Dependency closure prepared: %s agent(s), %s submolt(s), %s comment(s), %s vote(s)",
            len(self.agent_rows_by_source_id),
            len(self.submolt_rows_by_name),
            len(self.comment_rows_by_source_id),
            len(self.vote_rows_by_source_id),
        )

    def build_agent_mapping(self) -> None:
        for source_id in self.agent_rows_by_source_id:
            self.agent_id_map[source_id] = stable_uuid(AGENT_NAMESPACE, source_id)

    def build_submolt_mapping(self) -> None:
        for name in self.submolt_rows_by_name:
            self.submolt_id_map[name] = stable_uuid(SUBMOLT_NAMESPACE, name)

    def build_post_mapping(self) -> None:
        for source_id in self.post_rows_by_source_id:
            self.post_id_map[source_id] = stable_uuid(POST_NAMESPACE, source_id)

    def build_comment_mapping(self) -> None:
        for source_id in self.comment_rows_by_source_id:
            self.comment_id_map[source_id] = stable_uuid(COMMENT_NAMESPACE, source_id)

    def build_vote_mapping(self) -> None:
        for source_id in self.vote_rows_by_source_id:
            self.vote_id_map[source_id] = stable_uuid(VOTE_NAMESPACE, source_id)


    def ensure_agents(self, conn: psycopg.Connection) -> None:
        logger.info("Importing agents")
        with conn.cursor() as cur:
            for source_id, row in self.agent_rows_by_source_id.items():
                source_name = truncate_text(normalize_text(pick_value(row, "name", "agent_name", default=source_id), source_id), 128)
                existing = self._fetch_existing_id(cur, "agents", source_name, source_id)
                if existing:
                    self.agent_id_map[source_id] = existing
                    self.stats.skipped_agents += 1
                    continue

                agent_id = self.agent_id_map[source_id]
                cur.execute(
                    """
                    INSERT INTO agents (
                        id, name, display_name, description, avatar_url,
                        api_key_hash, claim_token, verification_code, status,
                        is_claimed, is_active, karma, follower_count, following_count,
                        owner_twitter_id, owner_twitter_handle, created_at, updated_at,
                        claimed_at, last_active
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (
                        agent_id,
                        source_name,
                        truncate_text(normalize_optional_text(pick_value(row, "display_name", default=source_name)) or source_name, 128),
                        normalize_optional_text(pick_value(row, "description", default="")),
                        normalize_optional_text(pick_value(row, "avatar_url")),
                        placeholder_hash("agent", source_id),
                        None,
                        None,
                        normalize_text(pick_value(row, "status", default="active"), "active"),
                        normalize_bool(pick_value(row, "is_claimed", default=False)),
                        True,
                        normalize_int(pick_value(row, "karma", default=0)),
                        normalize_int(pick_value(row, "follower_count", default=0)),
                        normalize_int(pick_value(row, "following_count", default=0)),
                        None,
                        normalize_optional_text(pick_value(row, "owner_x_handle", "owner_twitter_handle")),
                        timestamp_value(row, "created_at", "first_seen_at"),
                        timestamp_value(row, "updated_at", "last_seen_at", "created_at", "first_seen_at"),
                        None,
                        timestamp_value(row, "last_active", "last_seen_at", "created_at", "first_seen_at"),
                    ),
                )
                self.stats.inserted_agents += 1
        logger.info("Agents import complete: %s inserted, %s skipped", self.stats.inserted_agents, self.stats.skipped_agents)

    def ensure_submolts(self, conn: psycopg.Connection) -> None:
        logger.info("Importing submolts")
        with conn.cursor() as cur:
            for name, row in self.submolt_rows_by_name.items():
                name = truncate_text(name, 128)
                existing = self._fetch_existing_id(cur, "submolts", name, name)
                if existing:
                    self.submolt_id_map[name] = existing
                    self.stats.skipped_submolts += 1
                    continue

                submolt_id = self.submolt_id_map[name]
                creator_source_id = self.submolt_creator_sources.get(name)
                creator_id = self.agent_id_map.get(creator_source_id) if creator_source_id else None
                cur.execute(
                    """
                    INSERT INTO submolts (
                        id, name, display_name, description, avatar_url, banner_url,
                        banner_color, theme_color, subscriber_count, post_count,
                        creator_id, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s
                    )
                    """,
                    (
                        submolt_id,
                        name,
                        truncate_text(normalize_optional_text(pick_value(row, "display_name", default=name)) or name, 128),
                        normalize_optional_text(pick_value(row, "description", default="")),
                        normalize_optional_text(pick_value(row, "avatar_url")),
                        normalize_optional_text(pick_value(row, "banner_url")),
                        normalize_optional_text(pick_value(row, "banner_color")),
                        normalize_optional_text(pick_value(row, "theme_color")),
                        normalize_int(pick_value(row, "subscriber_count", default=0)),
                        normalize_int(pick_value(row, "post_count", default=0)),
                        creator_id,
                        timestamp_value(row, "created_at", "first_seen_at"),
                        timestamp_value(row, "updated_at", "created_at", "first_seen_at"),
                    ),
                )
                self.stats.inserted_submolts += 1
        logger.info("Submolts import complete: %s inserted, %s skipped", self.stats.inserted_submolts, self.stats.skipped_submolts)

    def ensure_posts(self, conn: psycopg.Connection) -> None:
        logger.info("Importing posts")
        with conn.cursor() as cur:
            for source_id, row in self.post_rows_by_source_id.items():
                existing = self._fetch_existing_id(cur, "posts", source_id)
                if existing:
                    self.post_id_map[source_id] = existing
                    self.stats.skipped_posts += 1
                    continue

                post_id = self.post_id_map[source_id]
                agent_source_id, _ = get_source_agent_key(row)
                submolt_name = get_source_submolt_name(row)
                agent_id = self.agent_id_map.get(agent_source_id or "") if agent_source_id else None
                submolt_id = self.submolt_id_map.get(submolt_name or "") if submolt_name else None

                if not agent_id or not submolt_id:
                    raise RuntimeError(f"Cannot import post {source_id}: missing agent or submolt mapping")

                content = normalize_optional_text(pick_value(row, "content", default=""))
                url = normalize_optional_text(pick_value(row, "url"))
                post_type = "link" if url else "text"
                title = normalize_text(pick_value(row, "title", default=""))
                if not title:
                    title = (content or url or source_id)[:300]

                cur.execute(
                    """
                    INSERT INTO posts (
                        id, author_id, submolt_id, submolt, title, content, url, post_type,
                        score, upvotes, downvotes, comment_count, is_pinned, is_deleted,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (
                        post_id,
                        agent_id,
                        submolt_id,
                        submolt_name,
                        title[:300],
                        content,
                        url,
                        post_type,
                        normalize_int(pick_value(row, "score", default=0)),
                        normalize_int(pick_value(row, "upvotes", default=0)),
                        normalize_int(pick_value(row, "downvotes", default=0)),
                        normalize_int(pick_value(row, "comment_count", default=0)),
                        normalize_bool(pick_value(row, "is_pinned", default=False)),
                        normalize_bool(pick_value(row, "is_deleted", default=False)),
                        timestamp_value(row, "created_at", "first_seen_at"),
                        timestamp_value(row, "updated_at", "created_at", "first_seen_at"),
                    ),
                )
                self.stats.inserted_posts += 1
        logger.info("Posts import complete: %s inserted, %s skipped", self.stats.inserted_posts, self.stats.skipped_posts)

    def ensure_comments(self, conn: psycopg.Connection) -> None:
        if not self.comment_rows_by_source_id:
            logger.info("No comments found for the sampled posts")
            return

        logger.info("Importing comments")
        depth_map = self._build_comment_depth_map()
        ordered_comments = sorted(
            self.comment_rows_by_source_id.items(),
            key=lambda item: (
                depth_map.get(item[0], 0),
                timestamp_value(item[1], "created_at") or "",
                item[0],
            ),
        )

        with conn.cursor() as cur:
            for source_id, row in ordered_comments:
                existing = self._fetch_existing_id(cur, "comments", source_id)
                if existing:
                    self.comment_id_map[source_id] = existing
                    self.stats.skipped_comments += 1
                    continue

                post_source_id = normalize_optional_text(pick_value(row, "post_id"))
                post_id = self.post_id_map.get(post_source_id or "")
                if not post_id:
                    continue

                agent_source_id, _ = get_source_agent_key(row)
                agent_id = self.agent_id_map.get(agent_source_id or "") if agent_source_id else None
                if not agent_id:
                    continue

                depth = depth_map.get(source_id, 0)
                if depth > 10:
                    continue

                parent_source_id = get_source_parent_id(row)
                parent_id = self.comment_id_map.get(parent_source_id or "") if parent_source_id else None

                comment_id = self.comment_id_map[source_id]
                content = normalize_text(pick_value(row, "content", default=""))
                if not content:
                    continue

                cur.execute(
                    """
                    INSERT INTO comments (
                        id, post_id, author_id, parent_id, content,
                        score, upvotes, downvotes, depth, is_deleted,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (
                        comment_id,
                        post_id,
                        agent_id,
                        parent_id,
                        content,
                        normalize_int(pick_value(row, "score", default=0)),
                        normalize_int(pick_value(row, "upvotes", default=0)),
                        normalize_int(pick_value(row, "downvotes", default=0)),
                        depth,
                        normalize_bool(pick_value(row, "is_deleted", default=False)),
                        timestamp_value(row, "created_at", "first_seen_at"),
                        timestamp_value(row, "updated_at", "created_at", "first_seen_at"),
                    ),
                )
                self.stats.inserted_comments += 1
        logger.info("Comments import complete: %s inserted, %s skipped", self.stats.inserted_comments, self.stats.skipped_comments)

    def ensure_votes(self, conn: psycopg.Connection) -> None:
        if not self.vote_rows_by_source_id:
            logger.info("No votes subset available; skipping vote import")
            return

        logger.info("Importing votes")
        with conn.cursor() as cur:
            for source_id, row in self.vote_rows_by_source_id.items():
                existing = self._fetch_existing_id(cur, "votes", source_id)
                if existing:
                    self.vote_id_map[source_id] = existing
                    self.stats.skipped_votes += 1
                    continue

                agent_source_id, _ = get_source_agent_key(row)
                agent_id = self.agent_id_map.get(agent_source_id or "") if agent_source_id else None
                if not agent_id:
                    continue

                target_source_id = get_target_id(row)
                target_type = normalize_text(pick_value(row, "target_type", default=""), "").lower()
                if target_type not in {"post", "comment"}:
                    if target_source_id in self.post_id_map:
                        target_type = "post"
                    elif target_source_id in self.comment_id_map:
                        target_type = "comment"
                    else:
                        continue

                target_id = self.post_id_map.get(target_source_id or "") if target_type == "post" else self.comment_id_map.get(target_source_id or "")
                if not target_id:
                    continue

                value = normalize_int(pick_value(row, "value", default=0))
                if value not in (-1, 1):
                    continue

                vote_id = self.vote_id_map[source_id]
                cur.execute(
                    """
                    INSERT INTO votes (
                        id, agent_id, target_id, target_type, value, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        vote_id,
                        agent_id,
                        target_id,
                        target_type,
                        value,
                        timestamp_value(row, "created_at", "first_seen_at"),
                    ),
                )
                self.stats.inserted_votes += 1
        logger.info("Votes import complete: %s inserted, %s skipped", self.stats.inserted_votes, self.stats.skipped_votes)

    def _build_comment_depth_map(self) -> dict[str, int]:
        depth_cache: dict[str, int] = {}

        def resolve_depth(comment_source_id: str, stack: set[str] | None = None) -> int:
            if comment_source_id in depth_cache:
                return depth_cache[comment_source_id]

            row = self.comment_rows_by_source_id.get(comment_source_id)
            if not row:
                depth_cache[comment_source_id] = 0
                return 0

            parent_source_id = get_source_parent_id(row)
            if not parent_source_id or parent_source_id not in self.comment_rows_by_source_id:
                depth_cache[comment_source_id] = 0
                return 0

            if stack is None:
                stack = set()
            if comment_source_id in stack:
                depth_cache[comment_source_id] = 0
                return 0

            stack.add(comment_source_id)
            depth = resolve_depth(parent_source_id, stack) + 1
            stack.remove(comment_source_id)
            depth_cache[comment_source_id] = depth
            return depth

        for comment_source_id in self.comment_rows_by_source_id:
            resolve_depth(comment_source_id)

        return depth_cache

    @staticmethod
    def _fetch_existing_id(cur: psycopg.Cursor[Any], table: str, *keys: str) -> str | None:
        if table in {"agents", "submolts"}:
            if not keys:
                return None

            # For agents and submolts, always look up by name
            name = keys[0]
            cur.execute(f"SELECT id FROM {table} WHERE name = %s", (name,))
            existing = cur.fetchone()
            return str(existing[0]) if existing else None

        source_id = keys[0] if keys else None
        if not source_id:
            return None

        cur.execute(f"SELECT id FROM {table} WHERE id = %s", (source_id,))
        existing = cur.fetchone()
        return str(existing[0]) if existing else None


def test_database_connection(database_url: str) -> None:
    logger.info("Testing PostgreSQL connection to: %s", database_url)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    logger.info("PostgreSQL connection test succeeded for: %s", database_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a random sample from the Moltbook Observatory HF archive into PostgreSQL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="HF split name")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of posts to sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling")
    parser.add_argument("--batch-size", type=int, default=500, help="Reserved for future batching")
    parser.add_argument("--posts-data-files", default=DEFAULT_POSTS_DATA_FILES, help="HF parquet glob for posts")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""), help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Only sample and summarize, do not write to the database")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_external_logging()
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Importer started")
    logger.info("Dataset: %s | Split: %s | Sample size: %s | Dry run: %s", args.dataset, args.split, args.count, args.dry_run)

    if args.count <= 0:
        print("Count must be greater than zero", file=sys.stderr)
        return 1

    if args.database_url:
        test_database_connection(args.database_url)
    elif not args.dry_run:
        print("DATABASE_URL is required", file=sys.stderr)
        return 1

    if args.dry_run:
        importer = ArchiveImporter(args.database_url or "", args.dataset, args.split, args.seed, args.batch_size, args.posts_data_files)
        sampled_posts = importer.sample_posts(args.count)
        print(f"Dry run: sampled {len(sampled_posts)} posts from {args.dataset}/{args.split}")
        logger.info("Dry run finished successfully")
        return 0

    importer = ArchiveImporter(args.database_url, args.dataset, args.split, args.seed, args.batch_size, args.posts_data_files)
    sampled_posts = importer.sample_posts(args.count)
    if not sampled_posts:
        print("No posts were sampled; nothing to import")
        return 0

    importer.collect_related_rows(sampled_posts)
    importer.build_agent_mapping()
    importer.build_submolt_mapping()
    importer.build_post_mapping()
    importer.build_comment_mapping()
    importer.build_vote_mapping()

    logger.info("Connecting to PostgreSQL database")
    with psycopg.connect(args.database_url) as conn:
        logger.info("Connected to PostgreSQL database successfully")
        importer.ensure_agents(conn)
        importer.ensure_submolts(conn)
        importer.ensure_posts(conn)
        importer.ensure_comments(conn)
        importer.ensure_votes(conn)

    print("Import complete")
    print(
        "Sampled posts: {sampled_posts}, agents inserted/skipped: {inserted_agents}/{skipped_agents}, "
        "submolts inserted/skipped: {inserted_submolts}/{skipped_submolts}, posts inserted/skipped: {inserted_posts}/{skipped_posts}, "
        "comments inserted/skipped: {inserted_comments}/{skipped_comments}, votes inserted/skipped: {inserted_votes}/{skipped_votes}".format(
            sampled_posts=importer.stats.sampled_posts,
            inserted_agents=importer.stats.inserted_agents,
            skipped_agents=importer.stats.skipped_agents,
            inserted_submolts=importer.stats.inserted_submolts,
            skipped_submolts=importer.stats.skipped_submolts,
            inserted_posts=importer.stats.inserted_posts,
            skipped_posts=importer.stats.skipped_posts,
            inserted_comments=importer.stats.inserted_comments,
            skipped_comments=importer.stats.skipped_comments,
            inserted_votes=importer.stats.inserted_votes,
            skipped_votes=importer.stats.skipped_votes,
        )
    )
    if not importer.vote_rows_by_source_id:
        print("Votes subset not available in the public archive; vote import was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())