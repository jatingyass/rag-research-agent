"""
backend/evaluation/ragas_eval.py

RAGAS evaluation pipeline — the feature that makes this portfolio project
stand out from 99% of RAG implementations.

Metrics evaluated:
  - Faithfulness: Does the answer contain only facts from the context?
  - Answer Relevancy: Is the answer relevant to the question?
  - Context Precision: Are the retrieved chunks relevant to the question?
  - Context Recall: Do the chunks cover all information needed to answer?

Uses Gemini as the judge LLM via LangchainLLMWrapper so no separate
OPENAI_API_KEY is required — the same Gemini key drives both the RAG
pipeline and the evaluation.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config import get_settings
from ..core.logging import get_logger
from ..agents.research_agent import get_research_agent

logger = get_logger("ragas_eval")


@dataclass
class TestCase:
    question: str
    ground_truth: str
    source_documents: list[str] | None = None


@dataclass
class EvalResult:
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_chunks: int

    @property
    def overall_score(self) -> float:
        return (
            self.faithfulness * 0.3
            + self.answer_relevancy * 0.3
            + self.context_precision * 0.2
            + self.context_recall * 0.2
        )

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "overall_score": round(self.overall_score, 3),
            "num_chunks": self.num_chunks,
        }


def _build_ragas_llm():
    """
    Return a RAGAS-compatible LLM wrapper backed by Gemini.
    Avoids the OpenAI dependency that RAGAS defaults to.
    """
    from ragas.llms import LangchainLLMWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    gemini = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.gemini_api_key,
    )
    return LangchainLLMWrapper(gemini)


def _build_ragas_embeddings():
    """Return RAGAS-compatible embeddings backed by the project's local model."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ..core.embeddings import get_embeddings
    return LangchainEmbeddingsWrapper(get_embeddings())


class RAGASEvaluator:
    """
    Runs RAGAS metrics on a set of test cases.
    Judge LLM: Gemini (same key as the RAG pipeline, no OpenAI required).
    """

    def __init__(self, user_id: str = "default"):
        self._settings = get_settings()
        self._agent = get_research_agent(user_id)

    def evaluate(self, test_cases: list[TestCase]) -> dict[str, Any]:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError("Run: pip install ragas datasets")

        judge_llm = _build_ragas_llm()
        judge_embeddings = _build_ragas_embeddings()

        # Inject Gemini into each metric
        for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
            metric.llm = judge_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = judge_embeddings

        questions, answers, contexts, ground_truths = [], [], [], []

        for tc in test_cases:
            logger.info(f"Evaluating: {tc.question[:60]}...")
            result = self._agent.query(tc.question)
            questions.append(tc.question)
            answers.append(result.answer)
            contexts.append(result.context_chunks)
            ground_truths.append(tc.ground_truth)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        ragas_result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        df = ragas_result.to_pandas()

        per_question = []
        for i, tc in enumerate(test_cases):
            per_question.append(EvalResult(
                question=tc.question,
                answer=answers[i],
                faithfulness=float(df["faithfulness"].iloc[i]),
                answer_relevancy=float(df["answer_relevancy"].iloc[i]),
                context_precision=float(df["context_precision"].iloc[i]),
                context_recall=float(df["context_recall"].iloc[i]),
                num_chunks=len(contexts[i]),
            ).to_dict())

        aggregate = {
            "faithfulness": round(float(df["faithfulness"].mean()), 3),
            "answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
            "context_precision": round(float(df["context_precision"].mean()), 3),
            "context_recall": round(float(df["context_recall"].mean()), 3),
            "num_test_cases": len(test_cases),
        }
        aggregate["overall_score"] = round(
            aggregate["faithfulness"] * 0.3
            + aggregate["answer_relevancy"] * 0.3
            + aggregate["context_precision"] * 0.2
            + aggregate["context_recall"] * 0.2,
            3,
        )

        return {"aggregate": aggregate, "per_question": per_question}

    def save_results(self, results: dict, output_path: str = "eval_results.json") -> None:
        Path(output_path).write_text(json.dumps(results, indent=2))
        logger.info(f"Evaluation results saved to {output_path}")


SAMPLE_TEST_CASES = [
    TestCase(
        question="What is retrieval-augmented generation and how does it work?",
        ground_truth=(
            "RAG is a technique that combines retrieval of relevant documents "
            "with language model generation to produce grounded, factual answers."
        ),
    ),
    TestCase(
        question="What are the main limitations of large language models?",
        ground_truth=(
            "LLMs suffer from hallucination, knowledge cutoffs, context window limits, "
            "high computational cost, and potential biases from training data."
        ),
    ),
]
