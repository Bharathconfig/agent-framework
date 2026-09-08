# Copyright (c) Microsoft. All rights reserved.

"""Tests for MCP-based skills (MCPSkillsSource, MCPSkill, MCPSkillResource)."""

from __future__ import annotations

import base64
import io
import json
import tarfile
import zipfile
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import (
    BlobResourceContents,
    ErrorData,
    ReadResourceResult,
    TextResourceContents,
)
from pydantic import AnyUrl

from agent_framework import MCPSkill, MCPSkillResource, MCPSkillsSource, SkillsSourceContext
from agent_framework._skills import _parse_mcp_skill_index

from .conftest import MockAgent

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


# Shared context for exercising skill sources where the agent/session are irrelevant.
_SOURCE_CTX = SkillsSourceContext(agent=MockAgent())  # type: ignore[abstract]  # pyrefly: ignore[bad-instantiation]

SAMPLE_SKILL_MD = """\
---
name: unit-converter
description: Convert between common units.
---
# Unit Converter

Body content here.
"""

SAMPLE_SKILL_INDEX = json.dumps({
    "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
    "skills": [
        {
            "name": "unit-converter",
            "type": "skill-md",
            "description": "Convert between common units.",
            "url": "skill://unit-converter/SKILL.md",
        }
    ],
})


def _make_text_result(text: str, uri: str = "skill://test") -> ReadResourceResult:
    """Create a ReadResourceResult with a single TextResourceContents."""
    return ReadResourceResult(contents=[TextResourceContents(uri=AnyUrl(uri), text=text, mimeType="text/markdown")])


def _make_blob_result(
    data: bytes,
    uri: str = "skill://test",
    mime_type: str = "application/octet-stream",
) -> ReadResourceResult:
    """Create a ReadResourceResult with a single BlobResourceContents."""
    return ReadResourceResult(
        contents=[BlobResourceContents(uri=AnyUrl(uri), blob=base64.b64encode(data).decode(), mimeType=mime_type)]
    )


def _make_empty_result() -> ReadResourceResult:
    """Create a ReadResourceResult with no contents."""
    return ReadResourceResult(contents=[])


def _make_client(**read_resource_responses: ReadResourceResult) -> AsyncMock:
    """Create a mock ClientSession whose read_resource returns different results per URI.

    Args:
        **read_resource_responses: Mapping of URI string to ReadResourceResult.
            Any URI not in this mapping raises McpError with the MCP-spec
            "Resource not found" code (-32002).
    """
    client = AsyncMock()

    async def _read_resource(uri: AnyUrl) -> ReadResourceResult:
        """Implements  read resource.
        
        Args:
            uri: Description of uri.
        
        Returns:
            Description of the return value.
        """
        uri_str = str(uri)
        if uri_str in read_resource_responses:
            return read_resource_responses[uri_str]
        raise McpError(error=ErrorData(code=-32002, message=f"Resource not found: {uri_str}"))

    client.read_resource = AsyncMock(side_effect=_read_resource)
    return client


# ---------------------------------------------------------------------------
# _parse_mcp_skill_index tests
# ---------------------------------------------------------------------------


class TestParseMCPSkillIndex:
    """Tests for the _parse_mcp_skill_index helper."""

    def test_parses_valid_index(self) -> None:
        """Validates behavior for parses valid index.
        
        Args:
            self: Description of self.
        """
        index = _parse_mcp_skill_index(SAMPLE_SKILL_INDEX)
        assert index.schema == "https://schemas.agentskills.io/discovery/0.2.0/schema.json"
        assert len(index.skills) == 1
        assert index.skills[0].name == "unit-converter"
        assert index.skills[0].type == "skill-md"
        assert index.skills[0].url == "skill://unit-converter/SKILL.md"

    def test_parses_empty_skills_array(self) -> None:
        """Validates behavior for parses empty skills array.
        
        Args:
            self: Description of self.
        """
        index = _parse_mcp_skill_index('{"$schema": "test", "skills": []}')
        assert index.skills == []

    def test_parses_missing_skills_key(self) -> None:
        """Validates behavior for parses missing skills key.
        
        Args:
            self: Description of self.
        """
        index = _parse_mcp_skill_index('{"$schema": "test"}')
        assert index.skills == []

    def test_raises_on_non_object(self) -> None:
        """Validates behavior for raises on non object.
        
        Args:
            self: Description of self.
        """
        with pytest.raises(ValueError, match="must be a JSON object"):
            _parse_mcp_skill_index("[]")

    def test_raises_on_invalid_json(self) -> None:
        """Validates behavior for raises on invalid json.
        
        Args:
            self: Description of self.
        """
        with pytest.raises(json.JSONDecodeError):
            _parse_mcp_skill_index("not json")

    def test_skips_non_dict_entries(self) -> None:
        """Validates behavior for skips non dict entries.
        
        Args:
            self: Description of self.
        """
        index = _parse_mcp_skill_index('{"skills": ["not-a-dict", {"name": "ok", "type": "skill-md"}]}')
        assert len(index.skills) == 1
        assert index.skills[0].name == "ok"


# ---------------------------------------------------------------------------
# MCPSkillResource tests
# ---------------------------------------------------------------------------


class TestMCPSkillsExperimentalStage:
    """Tests confirming the MCP skills types remain experimental (MCP_SKILLS)."""

    def test_docstrings_include_experimental_warning(self) -> None:
        """Validates behavior for docstrings include experimental warning.
        
        Args:
            self: Description of self.
        """
        assert MCPSkillResource.__doc__ is not None
        assert MCPSkill.__doc__ is not None
        assert MCPSkillsSource.__doc__ is not None

        assert ".. warning:: Experimental" in MCPSkillResource.__doc__
        assert ".. warning:: Experimental" in MCPSkill.__doc__
        assert ".. warning:: Experimental" in MCPSkillsSource.__doc__

    def test_feature_metadata_is_set(self) -> None:
        """Validates behavior for feature metadata is set.
        
        Args:
            self: Description of self.
        """
        for cls in (MCPSkillResource, MCPSkill, MCPSkillsSource):
            assert getattr(cls, "__feature_stage__", None) == "experimental"
            assert getattr(cls, "__feature_id__", None) == "MCP_SKILLS"


class TestMCPSkillResource:
    """Tests for MCPSkillResource."""

    async def test_read_text_content(self) -> None:
        """Validates behavior for read text content.
        
        Args:
            self: Description of self.
        """
        result = _make_text_result("hello world")
        resource = MCPSkillResource(name="test.md", result=result)
        content = await resource.read()
        assert content == "hello world"

    async def test_read_binary_content(self) -> None:
        """Validates behavior for read binary content.
        
        Args:
            self: Description of self.
        """
        data = bytes([0x01, 0x02, 0x03, 0x04])
        result = _make_blob_result(data)
        resource = MCPSkillResource(name="icon.bin", result=result)
        content = await resource.read()
        assert content == data

    async def test_read_empty_returns_none(self) -> None:
        """Validates behavior for read empty returns none.
        
        Args:
            self: Description of self.
        """
        result = _make_empty_result()
        resource = MCPSkillResource(name="empty", result=result)
        content = await resource.read()
        assert content is None

    async def test_read_multiple_text_contents_joined(self) -> None:
        """Validates behavior for read multiple text contents joined.
        
        Args:
            self: Description of self.
        """
        result = ReadResourceResult(
            contents=[
                TextResourceContents(uri=AnyUrl("skill://a"), text="line1", mimeType="text/plain"),
                TextResourceContents(uri=AnyUrl("skill://b"), text="line2", mimeType="text/plain"),
            ]
        )
        resource = MCPSkillResource(name="multi", result=result)
        content = await resource.read()
        assert content == "line1\nline2"

    async def test_binary_takes_precedence_over_text(self) -> None:
        """Validates behavior for binary takes precedence over text.
        
        Args:
            self: Description of self.
        """
        data = b"\xff\xfe"
        result = ReadResourceResult(
            contents=[
                TextResourceContents(uri=AnyUrl("skill://a"), text="text", mimeType="text/plain"),
                BlobResourceContents(
                    uri=AnyUrl("skill://b"),
                    blob=base64.b64encode(data).decode(),
                    mimeType="application/octet-stream",
                ),
            ]
        )
        resource = MCPSkillResource(name="mixed", result=result)
        content = await resource.read()
        # The implementation iterates all contents checking for BlobResourceContents
        # first, so when both text and binary are present, binary is returned.
        assert content == data


# ---------------------------------------------------------------------------
# MCPSkill tests
# ---------------------------------------------------------------------------


class TestMCPSkill:
    """Tests for MCPSkill."""

    async def test_get_content_fetches_and_caches(self) -> None:
        """Validates behavior for get content fetches and caches.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{"skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD)})
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://unit-converter/SKILL.md", client=client)

        content1 = await skill.get_content()
        content2 = await skill.get_content()

        assert "Body content here." in content1
        assert content1 == content2
        # Only one MCP call should be made (cached)
        assert client.read_resource.call_count == 1

    async def test_get_content_raises_on_empty(self) -> None:
        """Validates behavior for get content raises on empty.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{"skill://empty/SKILL.md": _make_empty_result()})
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="empty-skill", description="Empty skill.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://empty/SKILL.md", client=client)

        with pytest.raises(ValueError, match="no text content"):
            await skill.get_content()

    async def test_get_resource_text(self) -> None:
        """Validates behavior for get resource text.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{
            "skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD),
            "skill://unit-converter/references/checklist.md": _make_text_result("- check thing 1\n- check thing 2"),
        })
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://unit-converter/SKILL.md", client=client)

        resource = await skill.get_resource("references/checklist.md")
        assert resource is not None
        content = await resource.read()
        assert content == "- check thing 1\n- check thing 2"

    async def test_get_resource_binary(self) -> None:
        """Validates behavior for get resource binary.
        
        Args:
            self: Description of self.
        """
        data = bytes([0x01, 0x02, 0x03, 0x04])
        client = _make_client(**{
            "skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD),
            "skill://unit-converter/assets/icon.bin": _make_blob_result(data),
        })
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://unit-converter/SKILL.md", client=client)

        resource = await skill.get_resource("assets/icon.bin")
        assert resource is not None
        content = await resource.read()
        assert content == data

    async def test_get_resource_unknown_returns_none(self) -> None:
        """Validates behavior for get resource unknown returns none.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{"skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD)})
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://unit-converter/SKILL.md", client=client)

        resource = await skill.get_resource("references/does-not-exist.md")
        assert resource is None

    @pytest.mark.parametrize(
        "name",
        [
            "../escape.md",
            "references/../../escape.md",
            "..",
            "..\\escape.md",
            "/etc/passwd",
            "http://attacker.example.com/payload",
        ],
    )
    async def test_get_resource_path_traversal_returns_none(self, name: str) -> None:
        # Register a permissive mock that would happily return content for any URI,
        # so the test fails unless the client-side validation rejects the name
        # before issuing the read.
        """Validates behavior for get resource path traversal returns none.
        
        Args:
            self: Description of self.
            name: Description of name.
        """
        client = AsyncMock()
        client.read_resource = AsyncMock(return_value=_make_text_result("should never be returned"))

        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://unit-converter/SKILL.md", client=client)

        resource = await skill.get_resource(name)
        assert resource is None
        client.read_resource.assert_not_called()

    async def test_get_resource_empty_name_returns_none(self) -> None:
        """Validates behavior for get resource empty name returns none.
        
        Args:
            self: Description of self.
        """
        client = _make_client()
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)

        assert await skill.get_resource("") is None
        assert await skill.get_resource("   ") is None

    async def test_get_script_returns_none(self) -> None:
        """Validates behavior for get script returns none.
        
        Args:
            self: Description of self.
        """
        client = _make_client()
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)

        assert await skill.get_script("anything") is None

    def test_compute_skill_root_uri_strips_suffix(self) -> None:
        """Validates behavior for compute skill root uri strips suffix.
        
        Args:
            self: Description of self.
        """
        assert MCPSkill._compute_skill_root_uri("skill://unit-converter/SKILL.md") == "skill://unit-converter/"

    def test_compute_skill_root_uri_trailing_slash(self) -> None:
        """Validates behavior for compute skill root uri trailing slash.
        
        Args:
            self: Description of self.
        """
        assert MCPSkill._compute_skill_root_uri("skill://unit-converter/") == "skill://unit-converter/"

    def test_compute_skill_root_uri_no_suffix_adds_slash(self) -> None:
        """Validates behavior for compute skill root uri no suffix adds slash.
        
        Args:
            self: Description of self.
        """
        assert MCPSkill._compute_skill_root_uri("skill://unit-converter") == "skill://unit-converter/"

    async def test_session_provider_resolves_live_session(self) -> None:
        # A session_provider is resolved on every fetch, so a skill built against
        # one session follows a reconnect that swaps the session object.
        """Validates behavior for session provider resolves live session.
        
        Args:
            self: Description of self.
        """
        from agent_framework import SkillFrontmatter

        old_client = _make_client(**{"skill://unit-converter/SKILL.md": _make_text_result("# Old\nold body")})
        new_client = _make_client(**{"skill://unit-converter/SKILL.md": _make_text_result("# New\nnew body")})
        current = {"session": old_client}

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        skill = MCPSkill(
            frontmatter=fm,
            skill_md_uri="skill://unit-converter/SKILL.md",
            session_provider=lambda: current["session"],
        )

        # Swap the session (as a reconnect would) before the first fetch.
        current["session"] = new_client
        content = await skill.get_content()

        assert "new body" in content
        old_client.read_resource.assert_not_called()
        new_client.read_resource.assert_called_once()

    def test_requires_exactly_one_of_client_or_session_provider(self) -> None:
        """Validates behavior for requires exactly one of client or session provider.
        
        Args:
            self: Description of self.
        """
        from agent_framework import SkillFrontmatter

        fm = SkillFrontmatter(name="unit-converter", description="Convert between common units.")
        client = _make_client()

        with pytest.raises(ValueError, match="exactly one"):
            MCPSkill(frontmatter=fm, skill_md_uri="skill://x/SKILL.md")
        with pytest.raises(ValueError, match="exactly one"):
            MCPSkill(
                frontmatter=fm,
                skill_md_uri="skill://x/SKILL.md",
                client=client,
                session_provider=lambda: client,
            )


# ---------------------------------------------------------------------------
# MCPSkillsSource tests
# ---------------------------------------------------------------------------


class TestMCPSkillsSource:
    """Tests for MCPSkillsSource."""

    async def test_index_based_discovery_returns_skill(self) -> None:
        """Validates behavior for index based discovery returns skill.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{
            "skill://index.json": _make_text_result(SAMPLE_SKILL_INDEX, uri="skill://index.json"),
            "skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD),
        })
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)

        assert len(skills) == 1
        assert skills[0].frontmatter.name == "unit-converter"
        assert skills[0].frontmatter.description == "Convert between common units."

        # Content is fetched on demand, not during discovery
        content = await skills[0].get_content()
        assert "Body content here." in content

    async def test_no_index_returns_empty(self) -> None:
        """Validates behavior for no index returns empty.
        
        Args:
            self: Description of self.
        """
        client = _make_client()  # No resources at all
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_does_not_read_skill_md_during_discovery(self) -> None:
        # Index points to a skill, but SKILL.md is not registered on the server.
        # Discovery should succeed because it only reads the index.
        """Validates behavior for does not read skill md during discovery.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{"skill://index.json": _make_text_result(SAMPLE_SKILL_INDEX, uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)

        assert len(skills) == 1
        assert skills[0].frontmatter.name == "unit-converter"

    async def test_invalid_name_is_skipped(self) -> None:
        """Validates behavior for invalid name is skipped.
        
        Args:
            self: Description of self.
        """
        index_json = json.dumps({
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "name": "UnitConverter",  # Invalid: uppercase
                    "type": "skill-md",
                    "description": "Convert between common units.",
                    "url": "skill://UnitConverter/SKILL.md",
                }
            ],
        })
        client = _make_client(**{"skill://index.json": _make_text_result(index_json, uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_missing_required_fields_is_skipped(self) -> None:
        """Validates behavior for missing required fields is skipped.
        
        Args:
            self: Description of self.
        """
        index_json = json.dumps({
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "name": "unit-converter",
                    "type": "skill-md",
                    # Missing description and url
                }
            ],
        })
        client = _make_client(**{"skill://index.json": _make_text_result(index_json, uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    @pytest.mark.asyncio
    async def test_archive_missing_resource_is_skipped(self) -> None:
        # An archive entry whose archive resource is not available on the server
        # is skipped (the index is read, but the archive download fails).
        """Validates behavior for archive missing resource is skipped.
        
        Args:
            self: Description of self.
        """
        index_json = json.dumps({
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "name": "some-skill",
                    "type": "archive",
                    "description": "Packaged skill.",
                    "url": "skill://some-skill.tar.gz",
                }
            ],
        })
        client = _make_client(**{"skill://index.json": _make_text_result(index_json, uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_template_type_is_skipped(self) -> None:
        """Validates behavior for template type is skipped.
        
        Args:
            self: Description of self.
        """
        index_json = json.dumps({
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "type": "mcp-resource-template",
                    "description": "Per-product documentation skill",
                    "url": "skill://docs/{product}/SKILL.md",
                }
            ],
        })
        client = _make_client(**{"skill://index.json": _make_text_result(index_json, uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_empty_index_returns_empty(self) -> None:
        """Validates behavior for empty index returns empty.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{"skill://index.json": _make_text_result('{"skills": []}', uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_malformed_index_json_returns_empty(self) -> None:
        """Validates behavior for malformed index json returns empty.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{"skill://index.json": _make_text_result("not valid json", uri="skill://index.json")})
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_sibling_text_resource(self) -> None:
        """Validates behavior for sibling text resource.
        
        Args:
            self: Description of self.
        """
        client = _make_client(**{
            "skill://index.json": _make_text_result(SAMPLE_SKILL_INDEX, uri="skill://index.json"),
            "skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD),
            "skill://unit-converter/references/checklist.md": _make_text_result("- check thing 1\n- check thing 2"),
        })
        source = MCPSkillsSource(client=client)
        skill = (await source.get_skills(_SOURCE_CTX))[0]
        resource = await skill.get_resource("references/checklist.md")
        assert resource is not None
        content = await resource.read()
        assert content == "- check thing 1\n- check thing 2"

    async def test_sibling_binary_resource(self) -> None:
        """Validates behavior for sibling binary resource.
        
        Args:
            self: Description of self.
        """
        data = bytes([0x01, 0x02, 0x03, 0x04])
        client = _make_client(**{
            "skill://index.json": _make_text_result(SAMPLE_SKILL_INDEX, uri="skill://index.json"),
            "skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD),
            "skill://unit-converter/assets/icon.bin": _make_blob_result(data),
        })
        source = MCPSkillsSource(client=client)
        skill = (await source.get_skills(_SOURCE_CTX))[0]
        resource = await skill.get_resource("assets/icon.bin")
        assert resource is not None
        content = await resource.read()
        assert content == data

    async def test_session_provider_resolves_live_session(self) -> None:
        # Discovery and the resulting skills' on-demand fetches both resolve the
        # provider, so a source built before a reconnect follows the swapped session.
        """Validates behavior for session provider resolves live session.
        
        Args:
            self: Description of self.
        """
        old_client = _make_client(**{
            "skill://index.json": _make_text_result(SAMPLE_SKILL_INDEX, uri="skill://index.json"),
            "skill://unit-converter/SKILL.md": _make_text_result("# Old\nold body"),
        })
        new_client = _make_client(**{
            "skill://index.json": _make_text_result(SAMPLE_SKILL_INDEX, uri="skill://index.json"),
            "skill://unit-converter/SKILL.md": _make_text_result("# New\nnew body"),
        })
        current = {"session": old_client}

        source = MCPSkillsSource(session_provider=lambda: current["session"])
        skills = await source.get_skills(_SOURCE_CTX)
        assert len(skills) == 1

        # A reconnect swaps the session; the already-discovered skill must fetch
        # its content from the new session, not the closed one.
        current["session"] = new_client
        content = await skills[0].get_content()
        assert "new body" in content

    def test_requires_exactly_one_of_client_or_session_provider(self) -> None:
        """Validates behavior for requires exactly one of client or session provider.
        
        Args:
            self: Description of self.
        """
        client = _make_client()
        with pytest.raises(ValueError, match="exactly one"):
            MCPSkillsSource()
        with pytest.raises(ValueError, match="exactly one"):
            MCPSkillsSource(client=client, session_provider=lambda: client)


# ---------------------------------------------------------------------------
# McpError code branching tests
# ---------------------------------------------------------------------------


class TestMCPSkillsSourceErrorCodeBranching:
    """Tests that MCPSkillsSource and MCPSkill branch on McpError.error.code.

    Only "not found" codes (RESOURCE_NOT_FOUND -32002, METHOD_NOT_FOUND -32601)
    should be silently swallowed as "no skills available." Other McpError codes
    and non-McpError exceptions must propagate so that auth failures, server
    crashes, and connection drops are visible.
    """

    async def test_index_method_not_found_returns_empty(self) -> None:
        """METHOD_NOT_FOUND (-32601) -> server doesn't support resources/read."""
        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=McpError(error=ErrorData(code=-32601, message="Method not found")))
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_index_resource_not_found_returns_empty(self) -> None:
        """MCP-spec "Resource not found" (-32002) -> server has no index."""
        client = AsyncMock()
        client.read_resource = AsyncMock(
            side_effect=McpError(error=ErrorData(code=-32002, message="Resource not found"))
        )
        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_index_invalid_params_propagates(self) -> None:
        """INVALID_PARAMS (-32602) is a real bug, must propagate (not "not found")."""
        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=McpError(error=ErrorData(code=-32602, message="Invalid params")))
        source = MCPSkillsSource(client=client)
        with pytest.raises(McpError):
            await source.get_skills(_SOURCE_CTX)

    async def test_index_internal_error_propagates(self) -> None:
        """INTERNAL_ERROR (-32603) must propagate, not silently return empty."""
        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=McpError(error=ErrorData(code=-32603, message="Internal error")))
        source = MCPSkillsSource(client=client)
        with pytest.raises(McpError):
            await source.get_skills(_SOURCE_CTX)

    async def test_index_connection_closed_propagates(self) -> None:
        """CONNECTION_CLOSED (-32000) must propagate."""
        client = AsyncMock()
        client.read_resource = AsyncMock(
            side_effect=McpError(error=ErrorData(code=-32000, message="Connection closed"))
        )
        source = MCPSkillsSource(client=client)
        with pytest.raises(McpError):
            await source.get_skills(_SOURCE_CTX)

    async def test_index_generic_error_code_propagates(self) -> None:
        """Generic handler error (code 0) must propagate."""
        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=McpError(error=ErrorData(code=0, message="Some handler error")))
        source = MCPSkillsSource(client=client)
        with pytest.raises(McpError):
            await source.get_skills(_SOURCE_CTX)

    async def test_index_non_mcp_error_propagates(self) -> None:
        """Non-McpError exceptions (connection drop, timeout) must propagate."""
        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=ConnectionError("connection lost"))
        source = MCPSkillsSource(client=client)
        with pytest.raises(ConnectionError):
            await source.get_skills(_SOURCE_CTX)

    async def test_get_resource_internal_error_propagates(self) -> None:
        """McpError with INTERNAL_ERROR on get_resource must propagate."""
        from agent_framework import SkillFrontmatter

        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=McpError(error=ErrorData(code=-32603, message="Server crashed")))
        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)
        with pytest.raises(McpError):
            await skill.get_resource("references/file.md")

    async def test_get_resource_not_found_returns_none(self) -> None:
        """McpError with RESOURCE_NOT_FOUND (-32002) on get_resource returns None."""
        from agent_framework import SkillFrontmatter

        client = AsyncMock()
        client.read_resource = AsyncMock(
            side_effect=McpError(error=ErrorData(code=-32002, message="Resource not found"))
        )
        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)
        result = await skill.get_resource("references/file.md")
        assert result is None

    async def test_get_resource_connection_error_propagates(self) -> None:
        """A plain ConnectionError on get_resource must propagate, not return None."""
        from agent_framework import SkillFrontmatter

        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=ConnectionError("connection lost"))
        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)
        with pytest.raises(ConnectionError):
            await skill.get_resource("references/file.md")

    async def test_get_resource_timeout_error_propagates(self) -> None:
        """A TimeoutError on get_resource must propagate, not return None."""
        from agent_framework import SkillFrontmatter

        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=TimeoutError("read timed out"))
        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)
        with pytest.raises(TimeoutError):
            await skill.get_resource("references/file.md")

    async def test_get_resource_generic_mcp_error_propagates(self) -> None:
        """McpError with a generic code (0) on get_resource must propagate."""
        from agent_framework import SkillFrontmatter

        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=McpError(error=ErrorData(code=0, message="Handler error")))
        fm = SkillFrontmatter(name="test-skill", description="Test.")
        skill = MCPSkill(frontmatter=fm, skill_md_uri="skill://test/SKILL.md", client=client)
        with pytest.raises(McpError):
            await skill.get_resource("references/file.md")

    async def test_index_timeout_error_propagates(self) -> None:
        """A TimeoutError reading skill://index.json must propagate."""
        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=TimeoutError("read timed out"))
        source = MCPSkillsSource(client=client)
        with pytest.raises(TimeoutError):
            await source.get_skills(_SOURCE_CTX)


# ---------------------------------------------------------------------------
# Archive skill helpers
# ---------------------------------------------------------------------------


def _make_zip(files: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP archive from a ``{path: content}`` mapping."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _make_tar(files: dict[str, bytes], *, gzipped: bool) -> bytes:
    """Build an in-memory TAR (optionally gzip-compressed) archive."""
    buffer = io.BytesIO()

    def _write(archive: tarfile.TarFile) -> None:
        """Implements  write.
        
        Args:
            archive: Description of archive.
        """
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    if gzipped:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            _write(archive)
    else:
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            _write(archive)
    return buffer.getvalue()


ARCHIVE_SKILL_MD = """\
---
name: packaged-skill
description: A skill delivered as an archive.
---
# Packaged Skill

Instructions from an archive.
"""


def _make_archive_index(name: str, url: str, entry_type: str = "archive") -> str:
    """Build a skill index JSON document with a single archive entry."""
    return json.dumps({
        "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
        "skills": [
            {
                "name": name,
                "type": entry_type,
                "description": "A skill delivered as an archive.",
                "url": url,
            }
        ],
    })


def _archive_client(index_json: str, archive_url: str, archive_bytes: bytes, mime_type: str) -> AsyncMock:
    """Build a mock client that serves the index and a single archive blob resource."""
    return _make_client(**{
        "skill://index.json": _make_text_result(index_json, uri="skill://index.json"),
        str(AnyUrl(archive_url)): _make_blob_result(archive_bytes, uri=archive_url, mime_type=mime_type),
    })


# ---------------------------------------------------------------------------
# Archive skill discovery tests (through MCPSkillsSource)
# ---------------------------------------------------------------------------


class TestMCPSkillsSourceArchive:
    """Tests for archive-type skill discovery via MCPSkillsSource (in-memory)."""

    @pytest.mark.asyncio
    async def test_zip_archive_discovered_as_file_skill(self) -> None:
        """Validates behavior for zip archive discovered as file skill.
        
        Args:
            self: Description of self.
        """
        from agent_framework import FileSkill

        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({"SKILL.md": ARCHIVE_SKILL_MD.encode()})
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)

        assert len(skills) == 1
        skill = skills[0]
        assert isinstance(skill, FileSkill)
        assert skill.frontmatter.name == "packaged-skill"
        content = await skill.get_content()
        assert "Instructions from an archive." in content

    @pytest.mark.asyncio
    async def test_targz_archive_discovered(self) -> None:
        """Validates behavior for targz archive discovered.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.tar.gz"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_tar({"SKILL.md": ARCHIVE_SKILL_MD.encode()}, gzipped=True)
        client = _archive_client(index, url, archive, "application/gzip")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)

        assert len(skills) == 1
        assert skills[0].frontmatter.name == "packaged-skill"

    @pytest.mark.asyncio
    async def test_tar_archive_discovered(self) -> None:
        """Validates behavior for tar archive discovered.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.tar"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_tar({"SKILL.md": ARCHIVE_SKILL_MD.encode()}, gzipped=False)
        client = _archive_client(index, url, archive, "application/x-tar")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)

        assert len(skills) == 1
        assert skills[0].frontmatter.name == "packaged-skill"

    @pytest.mark.asyncio
    async def test_archive_reference_resource_is_readable(self) -> None:
        # A bundled reference file is served as an in-memory resource, read on demand.
        """Validates behavior for archive reference resource is readable.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({
            "SKILL.md": ARCHIVE_SKILL_MD.encode(),
            "references/refund-matrix.md": b"# Refund Matrix\nREF-CANARY-9001\n",
        })
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skill = (await source.get_skills(_SOURCE_CTX))[0]

        resource = await skill.get_resource("references/refund-matrix.md")
        assert resource is not None
        assert "REF-CANARY-9001" in await resource.read()

    @pytest.mark.asyncio
    async def test_wrapped_archive_root_is_discovered(self) -> None:
        # An archive whose SKILL.md sits under a top-level folder is still discovered,
        # and resources are resolved relative to the SKILL.md's directory.
        """Validates behavior for wrapped archive root is discovered.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({
            "packaged-skill/SKILL.md": ARCHIVE_SKILL_MD.encode(),
            "packaged-skill/references/doc.md": b"REF-CANARY-42\n",
        })
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skill = (await source.get_skills(_SOURCE_CTX))[0]

        assert skill.frontmatter.name == "packaged-skill"
        resource = await skill.get_resource("references/doc.md")
        assert resource is not None
        assert "REF-CANARY-42" in await resource.read()

    @pytest.mark.asyncio
    async def test_bundled_script_is_never_runnable(self) -> None:
        # An archive that bundles a .py script must not expose it as a runnable script,
        # nor (with default resource extensions) as a resource.
        """Validates behavior for bundled script is never runnable.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({
            "SKILL.md": ARCHIVE_SKILL_MD.encode(),
            "run.py": b"print('malicious')\n",
        })
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skill = (await source.get_skills(_SOURCE_CTX))[0]

        assert await skill.get_script("run.py") is None
        assert await skill.get_resource("run.py") is None
        content = await skill.get_content()
        assert "<available_scripts />" in content

    @pytest.mark.asyncio
    async def test_oversized_archive_download_is_skipped(self) -> None:
        """Validates behavior for oversized archive download is skipped.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({"SKILL.md": ARCHIVE_SKILL_MD.encode()})
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client, archive_max_size_bytes=8)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    @pytest.mark.asyncio
    async def test_archive_exceeding_file_count_is_skipped(self) -> None:
        """Validates behavior for archive exceeding file count is skipped.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({
            "SKILL.md": ARCHIVE_SKILL_MD.encode(),
            "a.md": b"a",
            "b.md": b"b",
        })
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client, archive_max_file_count=1)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    @pytest.mark.asyncio
    async def test_frontmatter_name_mismatch_is_skipped(self) -> None:
        # The SKILL.md frontmatter name must match the advertised entry name.
        """Validates behavior for frontmatter name mismatch is skipped.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        mismatched = ARCHIVE_SKILL_MD.replace("name: packaged-skill", "name: different-name")
        archive = _make_zip({"SKILL.md": mismatched.encode()})
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    async def test_frontmatter_name_mismatch_is_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Validates behavior for frontmatter name mismatch is logged as warning.
        
        Args:
            self: Description of self.
            caplog: Description of caplog.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        mismatched = ARCHIVE_SKILL_MD.replace("name: packaged-skill", "name: different-name")
        archive = _make_zip({"SKILL.md": mismatched.encode()})
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        with caplog.at_level("WARNING", logger="agent_framework._skills"):
            skills = await source.get_skills(_SOURCE_CTX)

        assert skills == []
        assert any(
            record.levelname == "WARNING" and "does not match the advertised entry name" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_archive_without_skill_md_is_skipped(self) -> None:
        """Validates behavior for archive without skill md is skipped.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({"readme.md": b"# not a skill\n"})
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    @pytest.mark.asyncio
    async def test_unsupported_archive_format_is_skipped(self) -> None:
        """Validates behavior for unsupported archive format is skipped.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.bin"
        index = _make_archive_index("packaged-skill", url)
        client = _archive_client(index, url, b"not-an-archive", "application/octet-stream")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []

    @pytest.mark.asyncio
    async def test_archive_download_internal_error_propagates(self) -> None:
        # A non-"not found" MCP error while downloading an archive must propagate,
        # not silently drop the skill (which would corrupt a CachingSkillsSource refresh).
        """Validates behavior for archive download internal error propagates.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)

        async def _read_resource(uri: AnyUrl) -> ReadResourceResult:
            """Implements  read resource.
            
            Args:
                uri: Description of uri.
            
            Returns:
                Description of the return value.
            """
            uri_str = str(uri)
            if uri_str == "skill://index.json":
                return _make_text_result(index, uri="skill://index.json")
            raise McpError(error=ErrorData(code=-32603, message="Internal error"))

        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=_read_resource)

        source = MCPSkillsSource(client=client)
        with pytest.raises(McpError):
            await source.get_skills(_SOURCE_CTX)

    @pytest.mark.asyncio
    async def test_archive_download_connection_error_propagates(self) -> None:
        # A plain ConnectionError while downloading an archive must propagate.
        """Validates behavior for archive download connection error propagates.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)

        async def _read_resource(uri: AnyUrl) -> ReadResourceResult:
            """Implements  read resource.
            
            Args:
                uri: Description of uri.
            
            Returns:
                Description of the return value.
            """
            uri_str = str(uri)
            if uri_str == "skill://index.json":
                return _make_text_result(index, uri="skill://index.json")
            raise ConnectionError("connection lost")

        client = AsyncMock()
        client.read_resource = AsyncMock(side_effect=_read_resource)

        source = MCPSkillsSource(client=client)
        with pytest.raises(ConnectionError):
            await source.get_skills(_SOURCE_CTX)

    @pytest.mark.asyncio
    async def test_mixed_skill_md_and_archive_entries(self) -> None:
        """Validates behavior for mixed skill md and archive entries.
        
        Args:
            self: Description of self.
        """
        archive_url = "skill://archives/packaged-skill.zip"
        index = json.dumps({
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "name": "unit-converter",
                    "type": "skill-md",
                    "description": "Convert between common units.",
                    "url": "skill://unit-converter/SKILL.md",
                },
                {
                    "name": "packaged-skill",
                    "type": "archive",
                    "description": "A skill delivered as an archive.",
                    "url": archive_url,
                },
            ],
        })
        archive = _make_zip({"SKILL.md": ARCHIVE_SKILL_MD.encode()})
        client = _make_client(**{
            "skill://index.json": _make_text_result(index, uri="skill://index.json"),
            "skill://unit-converter/SKILL.md": _make_text_result(SAMPLE_SKILL_MD),
            str(AnyUrl(archive_url)): _make_blob_result(archive, uri=archive_url, mime_type="application/zip"),
        })

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)

        names = sorted(s.frontmatter.name for s in skills)
        assert names == ["packaged-skill", "unit-converter"]

    @pytest.mark.asyncio
    async def test_zip_slip_archive_skips_whole_skill(self) -> None:
        # An archive with a path-traversal member is treated as hostile: the whole
        # skill is dropped (extraction raises, and _build_skill skips it).
        """Validates behavior for zip slip archive skips whole skill.
        
        Args:
            self: Description of self.
        """
        url = "skill://archives/packaged-skill.zip"
        index = _make_archive_index("packaged-skill", url)
        archive = _make_zip({
            "SKILL.md": ARCHIVE_SKILL_MD.encode(),
            "../evil.md": b"pwned",
        })
        client = _archive_client(index, url, archive, "application/zip")

        source = MCPSkillsSource(client=client)
        skills = await source.get_skills(_SOURCE_CTX)
        assert skills == []


# ---------------------------------------------------------------------------
# Archive extractor unit tests
# ---------------------------------------------------------------------------


class TestArchiveExtractor:
    """Tests for the archive format detection and hardened in-memory extraction helpers."""

    def test_detect_format_from_magic_bytes(self) -> None:
        """Validates behavior for detect format from magic bytes.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _detect_archive_format

        assert _detect_archive_format(b"\x1f\x8b\x08\x00", None, None) is _ArchiveFormat.TAR_GZ
        assert _detect_archive_format(b"PK\x03\x04rest", None, None) is _ArchiveFormat.ZIP

    def test_detect_format_from_media_type(self) -> None:
        """Validates behavior for detect format from media type.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _detect_archive_format

        assert _detect_archive_format(b"xx", "application/zip", None) is _ArchiveFormat.ZIP
        assert _detect_archive_format(b"xx", "application/x-tar", None) is _ArchiveFormat.TAR
        assert _detect_archive_format(b"xx", "application/gzip", None) is _ArchiveFormat.TAR_GZ

    def test_detect_format_from_url_suffix(self) -> None:
        """Validates behavior for detect format from url suffix.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _detect_archive_format

        assert _detect_archive_format(b"xx", None, "skill://a.zip") is _ArchiveFormat.ZIP
        assert _detect_archive_format(b"xx", None, "skill://a.tgz") is _ArchiveFormat.TAR_GZ
        assert _detect_archive_format(b"xx", None, "skill://a.tar") is _ArchiveFormat.TAR

    def test_detect_format_unknown(self) -> None:
        """Validates behavior for detect format unknown.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _detect_archive_format

        assert _detect_archive_format(b"xx", "text/plain", "skill://a.bin") is _ArchiveFormat.UNKNOWN

    def test_normalize_member_name_rejects_traversal(self) -> None:
        """Validates behavior for normalize member name rejects traversal.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _normalize_archive_member_name

        # Parent-traversal escapes raise (zip-slip is treated as a hostile archive).
        with pytest.raises(ValueError, match="escape"):
            _normalize_archive_member_name("../evil.md")
        with pytest.raises(ValueError, match="escape"):
            _normalize_archive_member_name("..\\evil.md")
        with pytest.raises(ValueError, match="escape"):
            _normalize_archive_member_name("a/../../evil.md")
        # Degenerate non-file entries that cannot escape are skipped (return None).
        assert _normalize_archive_member_name("") is None
        assert _normalize_archive_member_name("/") is None
        assert _normalize_archive_member_name(".") is None
        # A leading-slash path is neutralized to a relative path.
        assert _normalize_archive_member_name("/etc/passwd") == "etc/passwd"
        # Backslashes are normalized and redundant segments collapsed.
        assert _normalize_archive_member_name("refs\\./doc.md") == "refs/doc.md"
        assert _normalize_archive_member_name("ok/file.md") == "ok/file.md"

    def test_zip_slip_member_raises(self) -> None:
        """Validates behavior for zip slip member raises.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _extract_archive_to_memory

        # A member attempting a path-traversal escape aborts the whole extraction.
        archive = _make_zip({"safe.md": b"ok", "../evil.md": b"pwned"})
        with pytest.raises(ValueError, match="escape"):
            _extract_archive_to_memory(archive, _ArchiveFormat.ZIP, 20, 1024 * 1024)

    def test_leading_slash_member_is_neutralized(self) -> None:
        """Validates behavior for leading slash member is neutralized.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _extract_archive_to_memory

        archive = _make_zip({"/abs/file.md": b"data"})
        files = _extract_archive_to_memory(archive, _ArchiveFormat.ZIP, 20, 1024 * 1024)

        assert files == {"abs/file.md": b"data"}

    def test_file_count_limit_is_enforced(self) -> None:
        """Validates behavior for file count limit is enforced.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _extract_archive_to_memory

        archive = _make_zip({"a.md": b"a", "b.md": b"b", "c.md": b"c"})
        with pytest.raises(ValueError, match="file count"):
            _extract_archive_to_memory(archive, _ArchiveFormat.ZIP, 2, 1024 * 1024)

    def test_uncompressed_size_limit_is_enforced(self) -> None:
        """Validates behavior for uncompressed size limit is enforced.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _extract_archive_to_memory

        archive = _make_zip({"big.md": b"x" * 100})
        with pytest.raises(ValueError, match="uncompressed size"):
            _extract_archive_to_memory(archive, _ArchiveFormat.ZIP, 20, 10)

    def test_tar_symlink_member_is_skipped(self) -> None:
        """Validates behavior for tar symlink member is skipped.
        
        Args:
            self: Description of self.
        """
        from agent_framework._skills import _ArchiveFormat, _extract_archive_to_memory

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            link = tarfile.TarInfo(name="link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)
            data = b"regular"
            reg = tarfile.TarInfo(name="regular.md")
            reg.size = len(data)
            archive.addfile(reg, io.BytesIO(data))

        files = _extract_archive_to_memory(buffer.getvalue(), _ArchiveFormat.TAR, 20, 1024 * 1024)

        assert "link" not in files
        assert files == {"regular.md": b"regular"}
