import os
import json
import logging
import prompty
from opentelemetry import trace
from opentelemetry.trace import set_span_in_context
from azure.ai.evaluation import RelevanceEvaluator, GroundednessEvaluator, FluencyEvaluator, CoherenceEvaluator
from azure.ai.evaluation import ViolenceEvaluator, HateUnfairnessEvaluator, SelfHarmEvaluator, SexualEvaluator
from azure.ai.evaluation import evaluate
from azure.identity import DefaultAzureCredential

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData, ImageCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('promptflow').setLevel(logging.CRITICAL)
os.environ["PF_LOGGING_LEVEL"] = "CRITICAL"


class FriendlinessEvaluator:
    def __init__(self) -> None:
        pass

    def __call__(self, response):
        model_config = {
            "azure_deployment": os.environ.get("AZURE_OPENAI_4_EVAL_DEPLOYMENT_NAME"),
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT")
        }
        result = prompty.execute(
            "friendliness.prompty",
            configuration=model_config,
            inputs={
                "response": response,
            }
        )
        return {"score": result}


class ArticleEvaluator:
    def __init__(self, model_config, project_scope):
        self.evaluators = {
            "relevance": RelevanceEvaluator(model_config),
            "fluency": FluencyEvaluator(model_config),
            "coherence": CoherenceEvaluator(model_config),
            "groundedness": GroundednessEvaluator(model_config),
            "violence": ViolenceEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "hate_unfairness": HateUnfairnessEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "self_harm": SelfHarmEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "sexual": SexualEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "friendliness": FriendlinessEvaluator(),
        }
        self.project_scope = project_scope

    def __call__(self, *, data_path, **kwargs):
        output = {}
        result = evaluate(
            data=data_path,
            evaluators=self.evaluators,
            azure_ai_project=self.project_scope,
            evaluator_config={
                "relevance": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "fluency": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "coherence": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "query": "${data.query}",
                    },
                },
                "groundedness": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "violence": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "self_harm": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "hate_unfairness": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "sexual": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
                "friendliness": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "context": "${data.context}",
                        "query": "${data.query}",
                    },
                },
            },
        )
        output.update(result)
        return output


def evaluate_article(data, trace_context):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("run_evaluators", context=trace_context) as span:
        span.set_attribute("inputs", json.dumps(data))
        configuration = {
            "azure_deployment": os.environ["AZURE_OPENAI_4_EVAL_DEPLOYMENT_NAME"],
            "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
            "azure_endpoint": f"https://{os.getenv('AZURE_OPENAI_NAME')}.cognitiveservices.azure.com/"
        }
        project_scope = {
            "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
            "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
            "project_name": os.environ["AZURE_AI_PROJECT_NAME"],
        }
        evaluator = ArticleEvaluator(configuration, project_scope)
        results = evaluator(data)
        resultsJson = json.dumps(results)
        span.set_attribute("output", resultsJson)
        print("results: ", resultsJson)


def evaluate_image(image_path):
    endpoint = os.environ.get('CONTENT_SAFETY_ENDPOINT', "https://safety-ig.cognitiveservices.azure.com/")
    key = os.environ.get('CONTENT_SAFETY_KEY', "your_key_here")

    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    with open(image_path, "rb") as file:
        request = AnalyzeImageOptions(image=ImageData(content=file.read()))

    try:
        response = client.analyze_image(request)
    except HttpResponseError as e:
        print("Analyze image failed.")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
            raise
        raise

    hate_result = next((item for item in response.categories_analysis if item.category == ImageCategory.HATE), None)
    self_harm_result = next((item for item in response.categories_analysis if item.category == ImageCategory.SELF_HARM), None)
    sexual_result = next((item for item in response.categories_analysis if item.category == ImageCategory.SEXUAL), None)
    violence_result = next((item for item in response.categories_analysis if item.category == ImageCategory.VIOLENCE), None)

    results = [hate_result, self_harm_result, sexual_result, violence_result]

    unsafe_content = []

    for result in results:
        if result and result.severity > 0:
            unsafe_content.append({"category": str(result.category), "severity": str(result.severity)})

    if unsafe_content:
        return {"unsafe_content": unsafe_content}
    return {"unsafe_content": None}


class ImageEvaluator:
    def __init__(self, project_scope):
        from azure.ai.evaluation import (
            ViolenceMultimodalEvaluator,
            HateUnfairnessMultimodalEvaluator,
            SelfHarmMultimodalEvaluator,
            SexualMultimodalEvaluator,
            ProtectedMaterialMultimodalEvaluator,
        )
        self.evaluators = {
            "violence": ViolenceMultimodalEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "hate_unfairness": HateUnfairnessMultimodalEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "self_harm": SelfHarmMultimodalEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "sexual": SexualMultimodalEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
            "protected_material": ProtectedMaterialMultimodalEvaluator(azure_ai_project=project_scope, credential=DefaultAzureCredential()),
        }
        self.project_scope = project_scope

    def __call__(self, *, messages, **kwargs):
        output = {}
        data = [{"conversation": msg} for msg in messages]
        result = evaluate(
            data=data,
            evaluators=self.evaluators,
            azure_ai_project=self.project_scope,
        )
        output.update(result)
        return output


def evaluate_article_in_background(research_context, product_context, assignment_context, research, products, article):
    eval_data = {
        "query": json.dumps({
            "research_context": research_context,
            "product_context": product_context,
            "assignment_context": assignment_context
        }),
        "context": json.dumps({
            "research": research,
            "products": products,
        }),
        "response": json.dumps(article)
    }

    span = trace.get_current_span()
    trace_context = set_span_in_context(span)

    evaluate_article(eval_data, trace_context)
