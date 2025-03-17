# standard library
import asyncio
from datetime import datetime
import json
import random
from typing import Literal

# third-party libraries
import certifi
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

# local
from agents.rag_component import RAGModel
from utils.config import (
    BUSINESS_NAME,
    EMBEDDING_DIMENSIONS,
    INDEX_NAME,
    MONGO_GROUND_TRUTH_URI,
    OPENAI_EMBEDDING_MODEL_NAME,
    TIMEZONE,
)


class QA_pair(BaseModel):
    question: str = Field(description="The factoid question to be answered.")
    answer: str = Field(description="The answer to the factoid question.")


class GroundednessEval(BaseModel):
    evaluation: str = Field(description="your rationale for the rating, as a text")
    rating: str = Field(description="your rating, as a number between 0 and 7")


class RelevanceEval(BaseModel):
    evaluation: str = Field(description="your rationale for the rating, as a text")
    rating: str = Field(description="your rating, as a number between 0 and 7")


class StandAlonenessEval(BaseModel):
    evaluation: str = Field(description="your rationale for the rating, as a text")
    rating: str = Field(description="your rating, as a number between 0 and 7")


class AnswerEval(BaseModel):
    feedback: str = Field(description="your feedback for the criteria")
    rating: int = Field(description="your rating as a number between 0 and 7")


QA_GENERATION_PROMPT = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece \
of factual information from the context.
Your factoid question should be formulated in the same style as questions \
users could ask in a search engine.
This means that your factoid question MUST NOT mention something like \
"according to the passage" or "context".

Here is the context.

Context: {context}
"""


GROUNDEDNESS_CRITIQUE_PROMPT = """
You will be given a context and a question.
Your task is to provide a 'rating' scoring how well one can answer \
the given question unambiguously with the given context.
Give your answer on a scale of 0 to 7, where 0 means that the question is \
not answerable at all given the context, and 7 means that the question is \
clearly and unambiguously answerable with the context.

Here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """


RELEVANCE_CRITIQUE_PROMPT = """
You will be given a question.
Your task is to provide a 'rating' representing how useful this \
question can be to {BUSINESS_TYPE} guests.
Give your answer on a scale of 0 to 7, where 0 means that the question is \
not useful at all, and 7 means that the question is extremely useful.

Here is the question.

Question: {question}\n
Answer::: """


STANDALONENESS_CRITIQUE_PROMPT = """
You will be given a question.
Your task is to provide a 'rating' representing how context-independent \
this question is.
Give your answer on a scale of 0 to 7, where 0 means that the question \
depends on additional information to be understood, and 7 means that the \
question makes sense by itself.
For instance, if the question refers to a particular setting, like \
'in the context' or 'in the document', the rating must be 0.
The questions can contain obscure or specific information such as breakfast \
hours, luggage storage or ingredients of a dish and still be a 7: it must \
simply be clear to an operator with access to documentation what the question \
is about.

For instance, "And what about the other option?" should receive a 0, since \
there is an implicit mention of a context, thus the question is not independent \
from the context.

Here is the question.

Question: {question}\n
Answer::: """


ANSWER_EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, \
a reference answer that gets a score of 7, and a score rubric representing \
an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly \
based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 0 and 7. \
You should refer to the score rubric.
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 0: The response is completely incorrect, inaccurate, and/or not factual.
Score 1: The response is mostly incorrect, inaccurate, and/or not factual.
Score 2: The response is moderately incorrect, inaccurate, and/or not factual.
Score 3: The response is slighly incorrect, inaccurate, and/or not factual.
Score 4: The response is slighly correct, accurate, and/or factual.
Score 5: The response is moderately correct, accurate, and factual.
Score 6: The response is mostly correct, accurate, and factual.
Score 7: The response is completely correct, accurate, and factual.

###Feedback:"""


class TestSetGenerator:
    def __init__(
        self,
        docs: list[dict],
        N: int = 200,
        generate_new_questions: bool = True,
        evaluate_questions: bool = True,
        output_file_path: str = "./static/test_qa.json",
    ) -> None:
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.N_GENERATIONS = N
        self.docs = docs
        self._output_file_path = output_file_path
        self.qa_generator = (
            ChatPromptTemplate.from_template(QA_GENERATION_PROMPT)
            | self._llm.with_structured_output(QA_pair)
        )
        self.groundedness_critic = (
            ChatPromptTemplate.from_template(GROUNDEDNESS_CRITIQUE_PROMPT)
            | self._llm.with_structured_output(GroundednessEval)
        )
        self.standalone_critic = (
            ChatPromptTemplate.from_template(STANDALONENESS_CRITIQUE_PROMPT)
            | self._llm.with_structured_output(StandAlonenessEval)
        )
        self.relevance_critic = (
            ChatPromptTemplate.from_template(RELEVANCE_CRITIQUE_PROMPT)
            | self._llm.with_structured_output(RelevanceEval)
        )
        if generate_new_questions:
            self.eval_dataset = asyncio.run(self.generate_qa_pairs())
        else:
            with open(self._output_file_path, "r") as f:
                self.eval_dataset = json.load(f)
        if evaluate_questions:
            self.eval_dataset = asyncio.run(self.evaluate_questions())
        else:
            with open(self._output_file_path, "r") as f:
                self.eval_dataset = json.load(f)
        self.eval_dataset = self.select_good_questions()

    async def generate_qa_pairs(self, write_to_file: bool = True) -> list[dict]:
        print("Generating QA pairs...")
        output = []
        for sampled_context in tqdm(random.sample(self.docs, self.N_GENERATIONS)):
            qa_pair: QA_pair  = await self.qa_generator.ainvoke(
                {"context": sampled_context["text"]}
            )
            output.append(
                {
                    **qa_pair.model_dump(),
                    "context": sampled_context["text"],
                }
            )
        if write_to_file:
            with open(self._output_file_path, "w") as f:
                json.dump(output, f, indent=4)
        return output

    async def evaluate_questions(self, write_to_file: bool = True) -> list[dict]:
        print("Generating critique for each QA pair...")
        qas_evals = []
        for d in tqdm(self.eval_dataset):
            groundedness: GroundednessEval = await self.groundedness_critic.ainvoke(d)
            relevance: RelevanceEval = await self.relevance_critic.ainvoke(d)
            standalone: StandAlonenessEval = await self.standalone_critic.ainvoke(d)
            qas_evals.append(
                {
                    **d,
                    "groundedness_rating": groundedness.rating,
                    "relevance_rating": relevance.rating,
                    "standalone_rating": standalone.rating,
                    "groundedness_eval": groundedness.evaluation,
                    "relevance_eval": relevance.evaluation,
                    "standalone_eval": standalone.evaluation,
                }
            )
        if write_to_file:
            with open(self._output_file_path, "w") as f:
                json.dump(qas_evals, f, indent=4)
        return qas_evals

    def select_good_questions(self, data: list[dict]) -> list[dict]:
        previous = set()
        output = []
        for q in data:
            if (
                q["question"] not in previous
                and q["groundedness_rating"] >= 8
                and q["relevance_rating"] >= 8
                and q["standalone_rating"] >= 8
            ):
                previous.add(q["question"])
                output.append(q)
        return output


class RAGEvaluator:
    def __init__(self) -> None:
        self._client = MongoClient(
            MONGO_GROUND_TRUTH_URI,
            tlsCAFile=certifi.where()
        )
        self._db = self._client[BUSINESS_NAME.replace(" ", "")]
        self._collection = self._db[BUSINESS_NAME]
        self._vector_store = MongoDBAtlasVectorSearch(
            collection=self._collection,
            embedding=OpenAIEmbeddings(
                disallowed_special=(),
                model=OPENAI_EMBEDDING_MODEL_NAME,
                dimensions=EMBEDDING_DIMENSIONS,
            ),
            index_name=INDEX_NAME,
            relevance_score_fn="cosine",
        )
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.eval_chain = (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        "You are a fair evaluator language model."
                    ),
                    HumanMessagePromptTemplate.from_template(ANSWER_EVALUATION_PROMPT),
                ]
            )
            | self._llm.with_structured_output(AnswerEval)
        )
        self.rag_llm = RAGModel(vector_store=self._vector_store).agent_llm

    async def answer_with_rag(
        self,
        query: str,
        k: int = 3,
    ) -> tuple[str, list[str]]:
        retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "include_scores": True,
                "k": k,
            }
        )
        docs = await retriever.ainvoke(query)
        formatted_docs = [doc.page_content for doc in docs]
        response = await self.rag_llm.ainvoke(
            {
                "retrieved_context": "\n\n".join(formatted_docs),
                "input": query,
                "today": datetime.now(TIMEZONE).isoformat()
            }
        )
        return response, formatted_docs

    async def run_rag_tests(
        self,
        eval_set: list[dict],
        k: int = 3,
        verbose: bool = True,
        test_settings: str | None = None,
        output_file_path: str = "./static/rag_responses.json",
    ) -> list[dict]:
        output = []
        for example in tqdm(eval_set):
            question = example["question"]
            answer, relevant_docs = await self.answer_with_rag(question, k=k)
            if verbose:
                print("=======================================================")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f'True answer: {example["answer"]}')
            result = {
                "question": question,
                "true_answer": example["answer"],
                "generated_answer": answer,
                "retrieved_docs": relevant_docs,
            }
            if test_settings:
                result["test_settings"] = test_settings
            output.append(result)
        with open(output_file_path, "w") as f:
            json.dump(output, f, indent=4)
        return output

    async def evaluate_answers(
        self,
        eval_set: list[dict] | None = None,
        input_output_file_path: str | None = None,
    ) -> list[dict]:
        if eval_set is None and input_output_file_path is None:
            raise ValueError("Must provide either answers or input_file_path.")
        if eval_set is None:
            with open(input_output_file_path, "r") as f:
                eval_set = json.load(f)
        for experiment in tqdm(eval_set):
            eval_result: AnswerEval = await self.eval_chain.ainvoke(
                {
                    "instruction": experiment["question"],
                    "response": experiment["generated_answer"],
                    "reference_answer": experiment["true_answer"],
                }
            )
            experiment["eval_score"] = eval_result.rating
            experiment["eval_feedback"] = eval_result.feedback
        if input_output_file_path is not None:
            with open(input_output_file_path, "w") as f:
                json.dump(eval_set, f)
        return eval_set

    async def evaluate_rag(
        self,
        eval_set: list[dict],
        k: int = 3,
        llm: Literal["perplexity", "o1", "gpt-4o", "gpt-4o-mini"] = "perplexity",
        embeddings: Literal["voyageai", "openai"] = "openai",
        chunk_size: int = 200,
    ) -> None:
        settings_name = f"chunk:{chunk_size}_embeddings:{embeddings}_llm:{llm}"
        output_file_name = f"./static/rag_{settings_name}.json"
        print(f"Running evaluation for {settings_name}:")
        print("Running RAG...")
        responses = await self.run_rag_tests(
            eval_set,
            k=k,
            verbose=False,
            test_settings=settings_name,
            output_file_path=output_file_name,
        )
        print("Running evaluation...")
        await self.evaluate_answers(
            responses,
            output_file_name
        )
