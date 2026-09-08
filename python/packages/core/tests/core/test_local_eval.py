# Copyright (c) Microsoft. All rights reserved.

"""Tests for evaluator checks and LocalEvaluator."""

from __future__ import annotations

import inspect

import pytest

from agent_framework._evaluation import (
    CheckResult,
    EvalItem,
    EvalItemResult,
    EvalNotPassedError,
    EvalResults,
    EvalScoreResult,
    ExpectedToolCall,
    LocalEvaluator,
    RubricScore,
    _coerce_result,
    evaluator,
    keyword_check,
    tool_call_args_match,
    tool_called_check,
    tool_calls_present,
)
from agent_framework._types import Content, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    query: str = "What's the weather in Paris?",
    response: str = "It's sunny and 75°F",
    expected_output: str | None = None,
    conversation: list | None = None,
    tools: list | None = None,
    context: str | None = None,
) -> EvalItem:
    """Implements  make item.
    
    Args:
        query: Description of query.
        response: Description of response.
        expected_output: Description of expected_output.
        conversation: Description of conversation.
        tools: Description of tools.
        context: Description of context.
    
    Returns:
        Description of the return value.
    """
    if conversation is None:
        conversation = [Message("user", [query]), Message("assistant", [response])]
    return EvalItem(
        conversation=conversation,
        expected_output=expected_output,
        tools=tools,
        context=context,
    )


# ---------------------------------------------------------------------------
# Tier 1: (query, response) -> result
# ---------------------------------------------------------------------------


class TestTier1SimpleChecks:
    """Represents the TestTier1SimpleChecks type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_bool_return_true(self):
        """Validates behavior for bool return true.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def has_temperature(query: str, response: str) -> bool:
            """Implements has temperature.
            
            Args:
                query: Description of query.
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return "°F" in response

        result = await has_temperature(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True
        assert result.check_name == "has_temperature"

    @pytest.mark.asyncio
    async def test_bool_return_false(self):
        """Validates behavior for bool return false.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def has_celsius(query: str, response: str) -> bool:
            """Implements has celsius.
            
            Args:
                query: Description of query.
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return "°C" in response

        result = await has_celsius(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_float_return_passing(self):
        """Validates behavior for float return passing.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def length_score(response: str) -> float:
            """Implements length score.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return min(len(response) / 10, 1.0)

        result = await length_score(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True
        assert "score=" in result.reason

    @pytest.mark.asyncio
    async def test_float_return_failing(self):
        """Validates behavior for float return failing.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def always_low(response: str) -> float:
            """Implements always low.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return 0.1

        result = await always_low(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_response_only(self):
        """Function with only 'response' param should work."""

        @evaluator
        def is_short(response: str) -> bool:
            """Implements is short.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return len(response) < 1000

        result = await is_short(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_query_only(self):
        """Function with only 'query' param should work."""

        @evaluator
        def is_question(query: str) -> bool:
            """Implements is question.
            
            Args:
                query: Description of query.
            
            Returns:
                Description of the return value.
            """
            return "?" in query

        result = await is_question(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True


# ---------------------------------------------------------------------------
# Tier 2: (query, response, expected_output) -> result
# ---------------------------------------------------------------------------


class TestTier2GroundTruth:
    """Represents the TestTier2GroundTruth type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Validates behavior for exact match.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def exact_match(response: str, expected_output: str) -> bool:
            """Implements exact match.
            
            Args:
                response: Description of response.
                expected_output: Description of expected_output.
            
            Returns:
                Description of the return value.
            """
            return response.strip() == expected_output.strip()

        item = _make_item(response="42", expected_output="42")
        assert (await exact_match(item)).passed is True  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

        item2 = _make_item(response="43", expected_output="42")
        assert (await exact_match(item2)).passed is False  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

    @pytest.mark.asyncio
    async def test_expected_output_defaults_to_empty(self):
        """When expected_output is None on the item, it should be passed as ''."""

        @evaluator
        def check_expected(expected_output: str) -> bool:
            """Implements check expected.
            
            Args:
                expected_output: Description of expected_output.
            
            Returns:
                Description of the return value.
            """
            return expected_output == ""

        result = await check_expected(_make_item(expected_output=None))  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_similarity_score(self):
        """Validates behavior for similarity score.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def word_overlap(response: str, expected_output: str) -> float:
            """Implements word overlap.
            
            Args:
                response: Description of response.
                expected_output: Description of expected_output.
            
            Returns:
                Description of the return value.
            """
            r_words = set(response.lower().split())
            e_words = set(expected_output.lower().split())
            if not e_words:
                return 1.0
            return len(r_words & e_words) / len(e_words)

        item = _make_item(response="sunny warm day", expected_output="warm sunny afternoon")
        result = await word_overlap(item)  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True  # 2/3 overlap ≥ 0.5


# ---------------------------------------------------------------------------
# Tier 3: full context (conversation, tools, context)
# ---------------------------------------------------------------------------


class TestTier3FullContext:
    """Represents the TestTier3FullContext type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_conversation_access(self):
        """Validates behavior for conversation access.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def multi_turn(query: str, response: str, *, conversation: list) -> bool:
            """Implements multi turn.
            
            Args:
                query: Description of query.
                response: Description of response.
                conversation: Description of conversation.
            
            Returns:
                Description of the return value.
            """
            return len(conversation) >= 2

        item = _make_item(conversation=[Message("user", []), Message("assistant", [])])
        assert (await multi_turn(item)).passed is True  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

        item2 = _make_item(conversation=[Message("user", [])])
        assert (await multi_turn(item2)).passed is False  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

    @pytest.mark.asyncio
    async def test_tools_access(self):
        """Validates behavior for tools access.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def has_tools(tools: list) -> bool:
            """Implements has tools.
            
            Args:
                tools: Description of tools.
            
            Returns:
                Description of the return value.
            """
            return len(tools) > 0

        mock_tool = type(
            "MockTool",
            (),
            {"name": "get_weather", "description": "Get weather", "parameters": lambda self: {}},
        )()
        item = _make_item(tools=[mock_tool])
        assert (await has_tools(item)).passed is True  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

    @pytest.mark.asyncio
    async def test_context_access(self):
        """Validates behavior for context access.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def grounded(response: str, context: str) -> bool:
            """Implements grounded.
            
            Args:
                response: Description of response.
                context: Description of context.
            
            Returns:
                Description of the return value.
            """
            if not context:
                return True
            return any(word in response.lower() for word in context.lower().split())

        item = _make_item(response="It's sunny", context="sunny warm")
        assert (await grounded(item)).passed is True  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

    @pytest.mark.asyncio
    async def test_all_params(self):
        """Validates behavior for all params.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def full_check(
            query: str,
            response: str,
            expected_output: str,
            conversation: list,
            tools: list,
            context: str,
        ) -> bool:
            """Implements full check.
            
            Args:
                query: Description of query.
                response: Description of response.
                expected_output: Description of expected_output.
                conversation: Description of conversation.
                tools: Description of tools.
                context: Description of context.
            
            Returns:
                Description of the return value.
            """
            return all([query, response, expected_output is not None, isinstance(conversation, list)])

        item = _make_item(expected_output="foo", context="bar")
        assert (await full_check(item)).passed is True  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]


# ---------------------------------------------------------------------------
# Return type coercion
# ---------------------------------------------------------------------------


class TestReturnTypeCoercion:
    """Represents the TestReturnTypeCoercion type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_dict_with_score(self):
        """Validates behavior for dict with score.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def scored(response: str) -> dict:
            """Implements scored.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return {"score": 0.9, "reason": "good answer"}

        result = await scored(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True
        assert result.reason == "good answer"

    @pytest.mark.asyncio
    async def test_dict_with_score_below_threshold(self):
        """Validates behavior for dict with score below threshold.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def low_scored(response: str) -> dict:
            """Implements low scored.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return {"score": 0.3}

        result = await low_scored(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_dict_with_custom_threshold(self):
        """Validates behavior for dict with custom threshold.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def custom_threshold(response: str) -> dict:
            """Implements custom threshold.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return {"score": 0.3, "threshold": 0.2}

        result = await custom_threshold(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_dict_with_passed(self):
        """Validates behavior for dict with passed.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def explicit_pass(response: str) -> dict:
            """Implements explicit pass.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return {"passed": True, "reason": "all good"}

        result = await explicit_pass(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True
        assert result.reason == "all good"

    @pytest.mark.asyncio
    async def test_check_result_passthrough(self):
        """Validates behavior for check result passthrough.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def returns_check_result(response: str) -> CheckResult:
            """Implements returns check result.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return CheckResult(True, "direct result", "custom")

        result = await returns_check_result(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True
        assert result.reason == "direct result"
        assert result.check_name == "custom"

    @pytest.mark.asyncio
    async def test_unsupported_return_type(self):
        """Validates behavior for unsupported return type.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def bad_return(response: str) -> str:
            """Implements bad return.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return "oops"

        with pytest.raises(TypeError, match="unsupported type"):
            await bad_return(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

    @pytest.mark.asyncio
    async def test_int_return(self):
        """Validates behavior for int return.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def int_score(response: str) -> int:
            """Implements int score.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return 1

        result = await int_score(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True


# ---------------------------------------------------------------------------
# Decorator variants
# ---------------------------------------------------------------------------


class TestDecoratorVariants:
    """Represents the TestDecoratorVariants type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_decorator_no_parens(self):
        """Validates behavior for decorator no parens.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def my_check(response: str) -> bool:
            """Implements my check.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return True

        assert (await my_check(_make_item())).passed is True  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]

    @pytest.mark.asyncio
    async def test_decorator_with_name(self):
        """Validates behavior for decorator with name.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator(name="custom_name")
        def my_check(response: str) -> bool:
            """Implements my check.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return True

        assert my_check.__name__ == "custom_name"
        result = await my_check(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.check_name == "custom_name"

    @pytest.mark.asyncio
    async def test_direct_call(self):
        """Validates behavior for direct call.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        def raw_fn(query: str, response: str) -> bool:
            """Implements raw fn.
            
            Args:
                query: Description of query.
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return len(response) > 0

        check = evaluator(raw_fn, name="direct")  # type: ignore[call-overload]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
        result = await check(_make_item())
        assert result.passed is True
        assert result.check_name == "direct"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Represents the TestErrorHandling type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_unknown_required_param_raises(self):
        """Validates behavior for unknown required param raises.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        with pytest.raises(TypeError, match="unknown required parameter"):

            @evaluator
            def bad_params(query: str, unknown_param: str) -> bool:
                """Implements bad params.
                
                Args:
                    query: Description of query.
                    unknown_param: Description of unknown_param.
                
                Returns:
                    Description of the return value.
                """
                return True

    @pytest.mark.asyncio
    async def test_unknown_optional_param_ok(self):
        """Validates behavior for unknown optional param ok.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def optional_unknown(query: str, foo: str = "default") -> bool:
            """Implements optional unknown.
            
            Args:
                query: Description of query.
                foo: Description of foo.
            
            Returns:
                Description of the return value.
            """
            return foo == "default"

        result = await optional_unknown(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_async_function_works_with_evaluator(self):
        """Using an async function with @evaluator should work."""

        @evaluator
        async def async_fn(response: str) -> bool:
            """Implements async fn.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return True

        result = async_fn(_make_item())
        # Should return an awaitable
        assert inspect.isawaitable(result)
        check_result = await result
        assert check_result.passed is True  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Integration with LocalEvaluator
# ---------------------------------------------------------------------------


class TestLocalEvaluatorIntegration:
    """Represents the TestLocalEvaluatorIntegration type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_zero_checks(self):
        """A LocalEvaluator with no checks produces items with 0 scores, which count as failed."""
        local = LocalEvaluator()
        results = await local.evaluate([_make_item()])

        assert results.result_counts == {"passed": 0, "failed": 1, "errored": 0}
        assert results.all_passed is False
        assert results.items[0].scores == []
        with pytest.raises(EvalNotPassedError):
            results.raise_for_status()

    @pytest.mark.asyncio
    async def test_mixed_checks(self):
        """Function evaluators mix with built-in checks in LocalEvaluator."""

        @evaluator
        def length_ok(response: str) -> bool:
            """Implements length ok.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return len(response) > 5

        local = LocalEvaluator(
            keyword_check("sunny"),
            length_ok,
        )
        items = [_make_item()]
        results = await local.evaluate(items, eval_name="mixed test")

        assert results.status == "completed"
        assert results.result_counts["passed"] == 1  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]
        assert results.result_counts["failed"] == 0  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]

    @pytest.mark.asyncio
    async def test_evaluator_failure_counted(self):
        """Validates behavior for evaluator failure counted.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def always_fail(response: str) -> bool:
            """Implements always fail.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return False

        local = LocalEvaluator(always_fail)
        results = await local.evaluate([_make_item()])

        assert results.result_counts["failed"] == 1  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]

    @pytest.mark.asyncio
    async def test_multiple_evaluators(self):
        """Validates behavior for multiple evaluators.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def check_a(response: str) -> float:
            """Implements check a.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return 0.9

        @evaluator
        def check_b(query: str, response: str, expected_output: str) -> bool:
            """Implements check b.
            
            Args:
                query: Description of query.
                response: Description of response.
                expected_output: Description of expected_output.
            
            Returns:
                Description of the return value.
            """
            return True

        @evaluator(name="check_c")
        def check_c(response: str, conversation: list) -> dict:
            """Implements check c.
            
            Args:
                response: Description of response.
                conversation: Description of conversation.
            
            Returns:
                Description of the return value.
            """
            return {"score": 0.8, "reason": "looks good"}

        local = LocalEvaluator(check_a, check_b, check_c)
        results = await local.evaluate([_make_item(expected_output="test")])

        assert results.result_counts["passed"] == 1  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]
        assert "check_a" in results.per_evaluator
        assert "check_b" in results.per_evaluator
        assert "check_c" in results.per_evaluator


# ---------------------------------------------------------------------------
# Async evaluator (via @evaluator which handles async automatically)
# ---------------------------------------------------------------------------


class TestAsyncFunctionEvaluator:
    """Represents the TestAsyncFunctionEvaluator type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_async_evaluator_in_local(self):
        """Validates behavior for async evaluator in local.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        async def async_check(query: str, response: str) -> bool:
            """Implements async check.
            
            Args:
                query: Description of query.
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return len(response) > 0

        local = LocalEvaluator(async_check)
        results = await local.evaluate([_make_item()])
        assert results.result_counts["passed"] == 1  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]

    @pytest.mark.asyncio
    async def test_async_with_name(self):
        """Validates behavior for async with name.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator(name="named_async")
        async def my_async(response: str) -> float:
            """Implements my async.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return 0.75

        result = await my_async(_make_item())  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True
        assert result.check_name == "named_async"


# ---------------------------------------------------------------------------
# Auto-wrapping bare checks in evaluate_agent
# ---------------------------------------------------------------------------


class TestAutoWrapEvalChecks:
    """Represents the TestAutoWrapEvalChecks type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_bare_check_in_evaluators_list(self):
        """Bare EvalCheck callables are auto-wrapped in LocalEvaluator."""
        from agent_framework._evaluation import _run_evaluators

        @evaluator
        def is_long(response: str) -> bool:
            """Implements is long.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return len(response.split()) > 2

        items = [_make_item(response="It is sunny and warm today")]
        results = await _run_evaluators(is_long, items, eval_name="test")
        assert len(results) == 1
        assert results[0].result_counts["passed"] == 1  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]

    @pytest.mark.asyncio
    async def test_mixed_evaluators_and_checks(self):
        """Mix of Evaluator instances and bare checks works."""
        from agent_framework._evaluation import _run_evaluators

        @evaluator
        def has_words(response: str) -> bool:
            """Implements has words.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return len(response.split()) > 0

        local = LocalEvaluator(keyword_check("sunny"))

        items = [_make_item(response="It is sunny")]
        results = await _run_evaluators([local, has_words], items, eval_name="test")
        assert len(results) == 2
        assert all(r.result_counts["passed"] == 1 for r in results)  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]

    @pytest.mark.asyncio
    async def test_adjacent_checks_grouped(self):
        """Adjacent bare checks are grouped into a single LocalEvaluator."""
        from agent_framework._evaluation import _run_evaluators

        @evaluator
        def check_a(response: str) -> bool:
            """Implements check a.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return True

        @evaluator
        def check_b(response: str) -> bool:
            """Implements check b.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return True

        items = [_make_item()]
        results = await _run_evaluators([check_a, check_b], items, eval_name="test")
        # Two adjacent checks → one LocalEvaluator → one result
        assert len(results) == 1
        assert results[0].result_counts["passed"] == 1  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]


# ---------------------------------------------------------------------------
# Expected Tool Calls
# ---------------------------------------------------------------------------


def _make_tool_call_item(
    calls: list[tuple[str, dict | None]],
    expected: list[ExpectedToolCall] | None = None,
) -> EvalItem:
    """Build an EvalItem with tool calls in the conversation."""
    msgs: list[Message] = [Message("user", ["Do something"])]
    for name, args in calls:
        msgs.append(Message("assistant", [Content.from_function_call("call_" + name, name, arguments=args)]))
    msgs.append(Message("assistant", ["Done"]))
    return EvalItem(conversation=msgs, expected_tool_calls=expected)


class TestExpectedToolCallType:
    """Represents the TestExpectedToolCallType type and related behavior.
    """
    def test_name_only(self):
        """Validates behavior for name only.
        
        Args:
            self: Description of self.
        """
        tc = ExpectedToolCall("get_weather")
        assert tc.name == "get_weather"
        assert tc.arguments is None

    def test_name_and_args(self):
        """Validates behavior for name and args.
        
        Args:
            self: Description of self.
        """
        tc = ExpectedToolCall("get_weather", {"location": "NYC"})
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}


class TestToolCallsPresent:
    """Represents the TestToolCallsPresent type and related behavior.
    """
    def test_all_present(self):
        """Validates behavior for all present.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_weather", None), ("get_news", None)],
            expected=[ExpectedToolCall("get_weather"), ExpectedToolCall("get_news")],
        )
        result = tool_calls_present(item)
        assert result.passed is True
        assert result.check_name == "tool_calls_present"

    def test_missing_tool(self):
        """Validates behavior for missing tool.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_weather", None)],
            expected=[ExpectedToolCall("get_weather"), ExpectedToolCall("get_news")],
        )
        result = tool_calls_present(item)
        assert result.passed is False
        assert "get_news" in result.reason

    def test_extras_ok(self):
        """Validates behavior for extras ok.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_weather", None), ("get_news", None), ("get_stock", None)],
            expected=[ExpectedToolCall("get_weather")],
        )
        result = tool_calls_present(item)
        assert result.passed is True

    def test_no_expected(self):
        """Validates behavior for no expected.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(calls=[("get_weather", None)])
        result = tool_calls_present(item)
        assert result.passed is True
        assert "No expected" in result.reason


class TestToolCallArgsMatch:
    """Represents the TestToolCallArgsMatch type and related behavior.
    """
    def test_name_only_match(self):
        """Validates behavior for name only match.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_weather", {"location": "NYC"})],
            expected=[ExpectedToolCall("get_weather")],
        )
        result = tool_call_args_match(item)
        assert result.passed is True

    def test_args_exact_match(self):
        """Validates behavior for args exact match.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_weather", {"location": "NYC", "units": "fahrenheit"})],
            expected=[ExpectedToolCall("get_weather", {"location": "NYC"})],
        )
        # Subset match — extra "units" key is OK
        result = tool_call_args_match(item)
        assert result.passed is True

    def test_args_mismatch(self):
        """Validates behavior for args mismatch.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_weather", {"location": "LA"})],
            expected=[ExpectedToolCall("get_weather", {"location": "NYC"})],
        )
        result = tool_call_args_match(item)
        assert result.passed is False
        assert "args mismatch" in result.reason

    def test_tool_not_called(self):
        """Validates behavior for tool not called.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[("get_news", None)],
            expected=[ExpectedToolCall("get_weather", {"location": "NYC"})],
        )
        result = tool_call_args_match(item)
        assert result.passed is False
        assert "not called" in result.reason

    def test_multiple_expected(self):
        """Validates behavior for multiple expected.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(
            calls=[
                ("get_weather", {"location": "NYC"}),
                ("book_flight", {"destination": "LA", "date": "tomorrow"}),
            ],
            expected=[
                ExpectedToolCall("get_weather", {"location": "NYC"}),
                ExpectedToolCall("book_flight", {"destination": "LA"}),
            ],
        )
        result = tool_call_args_match(item)
        assert result.passed is True

    def test_no_expected(self):
        """Validates behavior for no expected.
        
        Args:
            self: Description of self.
        """
        item = _make_tool_call_item(calls=[("get_weather", None)])
        result = tool_call_args_match(item)
        assert result.passed is True


class TestExpectedToolCallsFieldInjection:
    """Test that @evaluator can receive expected_tool_calls via parameter injection."""

    @pytest.mark.asyncio
    async def test_injection(self):
        """Validates behavior for injection.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def check_tools(expected_tool_calls: list) -> bool:
            """Implements check tools.
            
            Args:
                expected_tool_calls: Description of expected_tool_calls.
            
            Returns:
                Description of the return value.
            """
            return len(expected_tool_calls) == 2

        item = _make_tool_call_item(
            calls=[],
            expected=[ExpectedToolCall("a"), ExpectedToolCall("b")],
        )
        result = await check_tools(item)  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_injection_empty_default(self):
        """Validates behavior for injection empty default.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def check_tools(expected_tool_calls: list) -> bool:
            """Implements check tools.
            
            Args:
                expected_tool_calls: Description of expected_tool_calls.
            
            Returns:
                Description of the return value.
            """
            return len(expected_tool_calls) == 0

        item = _make_tool_call_item(calls=[])
        result = await check_tools(item)  # type: ignore[misc]  # pyrefly: ignore[not-async]  # ty: ignore[invalid-await]
        assert result.passed is True


# ---------------------------------------------------------------------------
# Per-item results (auditing)
# ---------------------------------------------------------------------------


class TestPerItemResults:
    """LocalEvaluator should produce per-item EvalItemResult with query/response."""

    @pytest.mark.asyncio
    async def test_items_populated_with_query_and_response(self):
        """Validates behavior for items populated with query and response.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def is_sunny(response: str) -> bool:
            """Implements is sunny.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return "sunny" in response.lower()

        item = _make_item(query="Weather?", response="It's sunny!")
        local = LocalEvaluator(is_sunny)
        results = await local.evaluate([item])

        assert len(results.items) == 1
        ri = results.items[0]
        assert ri.item_id == "0"
        assert ri.status == "pass"
        assert ri.input_text == "Weather?"
        assert ri.output_text == "It's sunny!"
        assert len(ri.scores) == 1
        assert ri.scores[0].name == "is_sunny"
        assert ri.scores[0].passed is True

    @pytest.mark.asyncio
    async def test_items_populated_on_failure(self):
        """Validates behavior for items populated on failure.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def always_fail(response: str) -> bool:
            """Implements always fail.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return False

        item = _make_item(query="Hello", response="World")
        local = LocalEvaluator(always_fail)
        results = await local.evaluate([item])

        assert len(results.items) == 1
        ri = results.items[0]
        assert ri.status == "fail"
        assert ri.input_text == "Hello"
        assert ri.output_text == "World"
        assert ri.scores[0].passed is False
        assert ri.scores[0].score == 0.0

    @pytest.mark.asyncio
    async def test_multiple_items_indexed(self):
        """Validates behavior for multiple items indexed.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @evaluator
        def pass_all(response: str) -> bool:
            """Implements pass all.
            
            Args:
                response: Description of response.
            
            Returns:
                Description of the return value.
            """
            return True

        items = [
            _make_item(query="Q1", response="R1"),
            _make_item(query="Q2", response="R2"),
        ]
        local = LocalEvaluator(pass_all)
        results = await local.evaluate(items)

        assert len(results.items) == 2
        assert results.items[0].item_id == "0"
        assert results.items[0].input_text == "Q1"
        assert results.items[0].output_text == "R1"
        assert results.items[1].item_id == "1"
        assert results.items[1].input_text == "Q2"
        assert results.items[1].output_text == "R2"


# ---------------------------------------------------------------------------
# num_repetitions validation
# ---------------------------------------------------------------------------


class TestNumRepetitions:
    """Tests for the num_repetitions parameter on evaluate_agent."""

    @pytest.mark.asyncio
    async def test_num_repetitions_validation_rejects_zero(self):
        """Validates behavior for num repetitions validation rejects zero.
        
        Args:
            self: Description of self.
        """
        from agent_framework._evaluation import evaluate_agent

        with pytest.raises(ValueError, match="num_repetitions must be >= 1"):
            await evaluate_agent(
                queries=["Hello"],
                evaluators=LocalEvaluator(keyword_check("hello")),
                num_repetitions=0,
            )

    @pytest.mark.asyncio
    async def test_num_repetitions_validation_rejects_negative(self):
        """Validates behavior for num repetitions validation rejects negative.
        
        Args:
            self: Description of self.
        """
        from agent_framework._evaluation import evaluate_agent

        with pytest.raises(ValueError, match="num_repetitions must be >= 1"):
            await evaluate_agent(
                queries=["Hello"],
                evaluators=LocalEvaluator(keyword_check("hello")),
                num_repetitions=-1,
            )

    @pytest.mark.asyncio
    async def test_num_repetitions_multiplies_items(self):
        """num_repetitions=2 produces 2× the eval items."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_framework._evaluation import evaluate_agent
        from agent_framework._types import AgentResponse, Message

        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent.default_options = {}
        mock_agent.run = AsyncMock(return_value=AgentResponse(messages=[Message("assistant", ["reply"])]))

        results = await evaluate_agent(
            agent=mock_agent,
            queries=["Q1", "Q2"],
            evaluators=LocalEvaluator(keyword_check("reply")),
            num_repetitions=2,
        )
        # 2 queries × 2 reps = 4 items
        assert results[0].total == 4
        assert mock_agent.run.call_count == 4

    @pytest.mark.asyncio
    async def test_num_repetitions_with_expected_output(self):
        """num_repetitions > 1 correctly stamps expected_output via modulo."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_framework._evaluation import evaluate_agent
        from agent_framework._types import AgentResponse, Message

        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent.default_options = {}
        mock_agent.run = AsyncMock(return_value=AgentResponse(messages=[Message("assistant", ["reply"])]))

        @evaluator
        def check_expected(response: str, expected_output: str) -> dict:
            """Implements check expected.
            
            Args:
                response: Description of response.
                expected_output: Description of expected_output.
            
            Returns:
                Description of the return value.
            """
            return {"passed": expected_output in ("A", "B"), "reason": f"expected={expected_output}"}

        results = await evaluate_agent(
            agent=mock_agent,
            queries=["Q1", "Q2"],
            expected_output=["A", "B"],
            evaluators=LocalEvaluator(check_expected),
            num_repetitions=2,
        )
        # 2 queries × 2 reps = 4 items, all should pass
        assert results[0].total == 4
        assert results[0].passed == 4

    @pytest.mark.asyncio
    async def test_num_repetitions_with_expected_tool_calls(self):
        """num_repetitions > 1 correctly stamps expected_tool_calls via modulo."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_framework._evaluation import evaluate_agent
        from agent_framework._types import AgentResponse, Content, Message

        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent.default_options = {}
        mock_agent.run = AsyncMock(
            return_value=AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [Content.from_function_call("c1", "get_weather", arguments={"location": "NYC"})],
                    ),
                    Message("tool", [Content.from_function_result("c1", result="Sunny")]),
                    Message("assistant", ["It's sunny"]),
                ]
            )
        )

        results = await evaluate_agent(
            agent=mock_agent,
            queries=["Q1"],
            expected_tool_calls=[[ExpectedToolCall("get_weather")]],
            evaluators=LocalEvaluator(tool_calls_present),
            num_repetitions=2,
        )
        # 1 query × 2 reps = 2 items
        assert results[0].total == 2
        assert results[0].passed == 2


# ---------------------------------------------------------------------------
# r3 review: additional test coverage
# ---------------------------------------------------------------------------


class TestToolCalledCheckModeAny:
    """Tests for tool_called_check with mode='any'."""

    async def test_any_mode_one_tool_called(self):
        """mode='any' passes when at least one expected tool is called."""
        item = _make_item(
            conversation=[
                Message("user", ["Do something"]),
                Message("assistant", [Content.from_function_call("c1", "tool_a", arguments={})]),
                Message("tool", [Content.from_function_result("c1", result="ok")]),
                Message("assistant", ["Done"]),
            ]
        )
        check = tool_called_check("tool_a", "tool_b", mode="any")
        result = check(item)
        assert result.passed is True  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]

    async def test_any_mode_none_called(self):
        """mode='any' fails when no expected tools are called."""
        item = _make_item(
            conversation=[
                Message("user", ["Do something"]),
                Message("assistant", ["I can't use tools"]),
            ]
        )
        check = tool_called_check("tool_a", "tool_b", mode="any")
        result = check(item)
        assert result.passed is False  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        assert "None of expected tools" in result.reason  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]


class TestCoerceResultScoreError:
    """Tests for _coerce_result handling non-numeric score."""

    def test_non_numeric_score_raises(self):
        """Dict with non-numeric score raises TypeError."""
        with pytest.raises(TypeError, match="non-numeric 'score'"):
            _coerce_result({"score": "high"}, "test_check")

    def test_none_score_raises(self):
        """Validates behavior for none score raises.
        
        Args:
            self: Description of self.
        """
        with pytest.raises(TypeError, match="non-numeric 'score'"):
            _coerce_result({"score": None}, "test_check")


class TestBareCheckViaEvaluateAgent:
    """Test bare callable check functions through the public evaluate_agent API."""

    async def test_bare_check_through_evaluate_agent(self):
        """Validates behavior for bare check through evaluate agent.
        
        Args:
            self: Description of self.
        """
        from unittest.mock import AsyncMock, MagicMock

        from agent_framework._evaluation import evaluate_agent
        from agent_framework._types import AgentResponse

        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent.default_options = {}
        mock_agent.run = AsyncMock(
            return_value=AgentResponse(messages=[Message("assistant", ["The weather is sunny"])])
        )

        is_long = keyword_check("weather")

        results = await evaluate_agent(
            agent=mock_agent,
            queries=["Q"],
            evaluators=is_long,
        )
        assert results[0].total == 1
        assert results[0].passed == 1


class TestEvaluateAgentModuloWrapping:
    """Test that expected_output stamps correctly with num_repetitions > 1 and multiple queries."""

    async def test_modulo_stamps_correct_expected_output(self):
        """Validates behavior for modulo stamps correct expected output.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        from unittest.mock import AsyncMock, MagicMock

        from agent_framework._evaluation import evaluate_agent
        from agent_framework._types import AgentResponse

        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent.default_options = {}
        mock_agent.run = AsyncMock(return_value=AgentResponse(messages=[Message("assistant", ["reply"])]))

        # Track which expected_output each item gets
        seen_expected: list[str] = []

        @evaluator
        def capture_expected(response: str, expected_output: str) -> dict:
            """Implements capture expected.
            
            Args:
                response: Description of response.
                expected_output: Description of expected_output.
            
            Returns:
                Description of the return value.
            """
            seen_expected.append(expected_output)
            return {"passed": True, "reason": "ok"}

        await evaluate_agent(
            agent=mock_agent,
            queries=["Q1", "Q2", "Q3"],
            expected_output=["A", "B", "C"],
            evaluators=LocalEvaluator(capture_expected),
            num_repetitions=2,
        )
        # 3 queries × 2 reps = 6 items; modulo wrapping: A,B,C,A,B,C
        assert seen_expected == ["A", "B", "C", "A", "B", "C"]


class TestEvaluateAgentQueriesWithoutAgent:
    """Test error message when queries provided without agent."""

    async def test_queries_without_agent_gives_clear_error(self):
        """Validates behavior for queries without agent gives clear error.
        
        Args:
            self: Description of self.
        """
        from agent_framework._evaluation import evaluate_agent

        with pytest.raises(ValueError, match="Provide 'agent' when using 'queries'"):
            await evaluate_agent(
                queries=["hello"],
                evaluators=LocalEvaluator(keyword_check("x")),
            )


# ---------------------------------------------------------------------------
# r5 review: all_passed with result_counts=None + sub_results
# ---------------------------------------------------------------------------


class TestAllPassedSubResults:
    """Tests for EvalResults.all_passed with sub_results."""

    def test_all_passed_ignores_own_counts_when_none(self):
        """When result_counts is None (aggregate), all_passed delegates to sub_results."""
        from agent_framework._evaluation import EvalResults

        sub_pass = EvalResults(
            provider="Local",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 2, "failed": 0, "errored": 0},
        )
        parent = EvalResults(
            provider="Local",
            eval_id="e0",
            run_id="r0",
            status="completed",
            result_counts=None,
            sub_results={"agent1": sub_pass},
        )
        assert parent.all_passed is True

    def test_all_passed_parent_fails_when_own_counts_fail(self):
        """When parent has result_counts with failures, all_passed is False even if sub_results pass."""
        from agent_framework._evaluation import EvalResults

        sub_pass = EvalResults(
            provider="Local",
            eval_id="e1",
            run_id="r1",
            status="completed",
            result_counts={"passed": 2, "failed": 0, "errored": 0},
        )
        parent = EvalResults(
            provider="Local",
            eval_id="e0",
            run_id="r0",
            status="completed",
            result_counts={"passed": 1, "failed": 1, "errored": 0},
            sub_results={"agent1": sub_pass},
        )
        assert parent.all_passed is False


# ---------------------------------------------------------------------------
# Rubric assertions (EvalResults.assert_*)
# ---------------------------------------------------------------------------


def _rubric_results(*scores_per_item: list[EvalScoreResult]) -> EvalResults:
    """Implements  rubric results.
    
    Args:
        *scores_per_item: Description of scores_per_item.
    
    Returns:
        Description of the return value.
    """
    items = [
        EvalItemResult(item_id=f"item-{i}", status="pass", scores=scores) for i, scores in enumerate(scores_per_item)
    ]
    return EvalResults(
        provider="test",
        eval_id="ev1",
        run_id="run1",
        result_counts={"passed": len(items), "failed": 0, "errored": 0, "total": len(items)},
        items=items,
    )


class TestRubricAssertions:
    """Tests for EvalResults.assert_dimension_score_at_least."""

    def test_dimension_at_or_above_threshold_passes(self) -> None:
        """Validates behavior for dimension at or above threshold passes.
        
        Args:
            self: Description of self.
        """
        results = _rubric_results(
            [
                EvalScoreResult(
                    name="policy",
                    score=0.9,
                    dimensions=[RubricScore(id="clarity", score=4, applicable=True, weight=1, reason="")],
                )
            ],
        )
        # Should not raise.
        results.assert_dimension_score_at_least("clarity", 3)

    def test_dimension_below_threshold_raises(self) -> None:
        """Validates behavior for dimension below threshold raises.
        
        Args:
            self: Description of self.
        """
        results = _rubric_results(
            [
                EvalScoreResult(
                    name="policy",
                    score=0.5,
                    dimensions=[RubricScore(id="clarity", score=2, applicable=True, weight=1, reason="")],
                )
            ],
        )
        with pytest.raises(EvalNotPassedError):
            results.assert_dimension_score_at_least("clarity", 3)

    def test_non_applicable_skipped_by_default(self) -> None:
        """Validates behavior for non applicable skipped by default.
        
        Args:
            self: Description of self.
        """
        results = _rubric_results(
            [
                EvalScoreResult(
                    name="policy",
                    score=1.0,
                    dimensions=[RubricScore(id="clarity", score=None, applicable=False, weight=1, reason="n/a")],
                )
            ],
        )
        # No applicable scores; default behaviour is to skip silently.
        results.assert_dimension_score_at_least("clarity", 3)

    def test_require_applicable_raises_when_dimension_absent(self) -> None:
        """Validates behavior for require applicable raises when dimension absent.
        
        Args:
            self: Description of self.
        """
        results = _rubric_results(
            [EvalScoreResult(name="policy", score=1.0, dimensions=[])],
        )
        with pytest.raises(EvalNotPassedError, match="not applicable"):
            results.assert_dimension_score_at_least("clarity", 3, require_applicable=True)

    def test_require_applicable_raises_when_filtered_evaluator_missing(self) -> None:
        # Regression: previously the (not evaluator or found_any) guard caused
        # this case to silently pass even with require_applicable=True.
        """Validates behavior for require applicable raises when filtered evaluator missing.
        
        Args:
            self: Description of self.
        """
        results = _rubric_results(
            [
                EvalScoreResult(
                    name="other",
                    score=0.9,
                    dimensions=[RubricScore(id="clarity", score=4, applicable=True, weight=1, reason="")],
                )
            ],
        )
        with pytest.raises(EvalNotPassedError, match="not applicable"):
            results.assert_dimension_score_at_least("clarity", 3, evaluator="policy", require_applicable=True)

    def test_evaluator_filter_isolates_offenders(self) -> None:
        """Validates behavior for evaluator filter isolates offenders.
        
        Args:
            self: Description of self.
        """
        results = _rubric_results(
            [
                EvalScoreResult(
                    name="other",
                    score=0.1,
                    dimensions=[RubricScore(id="clarity", score=1, applicable=True, weight=1, reason="")],
                ),
                EvalScoreResult(
                    name="policy",
                    score=0.9,
                    dimensions=[RubricScore(id="clarity", score=4, applicable=True, weight=1, reason="")],
                ),
            ],
        )
        # The low-scoring "other" evaluator is filtered out; "policy" passes.
        results.assert_dimension_score_at_least("clarity", 3, evaluator="policy")


def _score_results(
    *scores_per_item: list[EvalScoreResult],
    sub_results: dict[str, EvalResults] | None = None,
) -> EvalResults:
    """Build an EvalResults shaped for score / status assertion tests."""
    items = [
        EvalItemResult(item_id=f"item-{i}", status="pass", scores=scores) for i, scores in enumerate(scores_per_item)
    ]
    return EvalResults(
        provider="test",
        eval_id="ev1",
        run_id="run1",
        result_counts={"passed": len(items), "failed": 0, "errored": 0, "total": len(items)},
        items=items,
        sub_results=sub_results or {},
    )


class TestAssertScoreAtLeast:
    """Tests for EvalResults.assert_score_at_least (mirrors .NET coverage)."""

    def test_all_above_threshold_passes(self) -> None:
        """Validates behavior for all above threshold passes.
        
        Args:
            self: Description of self.
        """
        results = _score_results(
            [EvalScoreResult(name="relevance", score=0.9)],
            [EvalScoreResult(name="relevance", score=0.85)],
        )
        # Should not raise.
        results.assert_score_at_least(0.8)

    def test_below_threshold_raises_with_offenders(self) -> None:
        """Validates behavior for below threshold raises with offenders.
        
        Args:
            self: Description of self.
        """
        results = _score_results(
            [EvalScoreResult(name="relevance", score=0.4)],
            [EvalScoreResult(name="relevance", score=0.9)],
        )
        with pytest.raises(EvalNotPassedError) as exc:
            results.assert_score_at_least(0.5)
        msg = str(exc.value)
        assert "item-0" in msg
        assert "relevance" in msg
        assert "0.400" in msg

    def test_evaluator_filter_isolates_offenders(self) -> None:
        """Validates behavior for evaluator filter isolates offenders.
        
        Args:
            self: Description of self.
        """
        results = _score_results(
            [
                EvalScoreResult(name="other", score=0.1),
                EvalScoreResult(name="relevance", score=0.95),
            ],
        )
        # The low-scoring "other" evaluator is filtered out; "relevance" passes.
        results.assert_score_at_least(0.8, evaluator="relevance")

    def test_recursion_into_sub_results(self) -> None:
        """Validates behavior for recursion into sub results.
        
        Args:
            self: Description of self.
        """
        sub = _score_results([EvalScoreResult(name="relevance", score=0.2)])
        parent = _score_results(
            [EvalScoreResult(name="relevance", score=0.9)],
            sub_results={"sub_executor": sub},
        )
        with pytest.raises(EvalNotPassedError) as exc:
            parent.assert_score_at_least(0.5)
        # Offender from sub-result is surfaced.
        assert "0.200" in str(exc.value)


class TestAssertNoFailedItems:
    """Tests for EvalResults.assert_no_failed_items (mirrors .NET coverage)."""

    def test_all_passing_does_not_raise(self) -> None:
        """Validates behavior for all passing does not raise.
        
        Args:
            self: Description of self.
        """
        results = _score_results(
            [EvalScoreResult(name="relevance", score=0.9)],
            [EvalScoreResult(name="relevance", score=0.85)],
        )
        # Should not raise.
        results.assert_no_failed_items()

    def test_failed_and_errored_items_raise_with_statuses(self) -> None:
        """Validates behavior for failed and errored items raise with statuses.
        
        Args:
            self: Description of self.
        """
        items = [
            EvalItemResult(item_id="ok", status="pass", scores=[]),
            EvalItemResult(item_id="bad", status="fail", scores=[]),
            EvalItemResult(item_id="boom", status="error", scores=[], error_code="timeout"),
        ]
        results = EvalResults(
            provider="test",
            eval_id="ev1",
            run_id="run1",
            result_counts={"passed": 1, "failed": 1, "errored": 1, "total": 3},
            items=items,
        )
        with pytest.raises(EvalNotPassedError) as exc:
            results.assert_no_failed_items()
        msg = str(exc.value)
        assert "bad:fail" in msg
        assert "boom:error" in msg

    def test_recursion_into_sub_results(self) -> None:
        """Validates behavior for recursion into sub results.
        
        Args:
            self: Description of self.
        """
        sub_items = [EvalItemResult(item_id="sub-bad", status="fail", scores=[])]
        sub = EvalResults(
            provider="test",
            eval_id="ev2",
            run_id="run2",
            result_counts={"passed": 0, "failed": 1, "errored": 0, "total": 1},
            items=sub_items,
        )
        parent = _score_results(
            [EvalScoreResult(name="relevance", score=0.9)],
            sub_results={"sub_executor": sub},
        )
        with pytest.raises(EvalNotPassedError) as exc:
            parent.assert_no_failed_items()
        assert "sub-bad:fail" in str(exc.value)
