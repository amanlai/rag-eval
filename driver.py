from rag_evaluation import RAGEvaluator, TestSetGenerator

def driver():
    rag_evaluator = RAGEvaluator()
    docs = list(rag_evaluator._collection.find())
    eval_set = TestSetGenerator(docs=docs).eval_dataset
    rag_evaluator.evaluate_rag(eval_set)
    rag_evaluator._client.close()


if __name__ == "__main__":
    driver()
