import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  
from utils.llm_init import LLM_initializer

@tool
def fluency_rating(speech: str) -> str:
    """
    Evaluate a user's speech fluency and coherence according to IELTS Speaking criteria.

    Current behavior:
    - Returns a placeholder string since the tool is still under development.

    Expected future behavior:
    - Analyze speech rhythm, pauses, hesitation, flow of ideas, and logical structuring.
    - Return a fluency band score (0–9) with justification.
    """
    return "This tool is under development"


@tool
def vocabulary_rating(speech: str) -> str:
    """
    Evaluate the user's lexical resource (vocabulary usage) based on IELTS Speaking criteria.

    Current behavior:
    - Returns a placeholder string since the tool is still under development.

    Expected future behavior:
    - Assess range of vocabulary, accuracy, collocations, paraphrasing ability, and topic-specific terms.
    - Return a vocabulary band score (0–9) with justification.
    """
    return "This tool is under development"


@tool
def grammatical_rating(speech: str) -> str:
    """
    Evaluate the user's grammatical range and accuracy according to IELTS Speaking scoring guidelines.

    Current behavior:
    - Returns a placeholder string since the tool is still under development.

    Expected future behavior:
    - Analyze sentence structures, tense consistency, syntax correctness, and error frequency.
    - Return a grammar band score (0–9) with justification.
    """
    return "This tool is under development"


@tool
def pronunciation_rating(speech: str) -> str:
    """
    Evaluate the user's pronunciation based on IELTS Speaking rubric.

    Current behavior:
    - Returns a placeholder string since the tool is still under development.

    Expected future behavior:
    - Examine clarity, stress, intonation, connected speech, and comprehensibility.
    - Return a pronunciation band score (0–9) with justification.
    """
    return "This tool is under development"


# @tool
def evaluator(speech: str) -> str:
    """
    Main IELTS speaking evaluator tool.

    Args:
        speech (str): A transcription of the candidate's spoken response.
                      (Timestamps or filler indications are allowed.)

    Behavior:
    - Creates an evaluation agent with IELTS scoring rubric.
    - Calls LLM to produce band scores and justification.

    Returns:
        dict / str: Agent response containing band scores and evaluation summary.

    Notes:
    - Uses placeholder scoring tools for now.
    - Will integrate real modular scoring engines later (fluency, vocabulary, grammar, pronunciation).
    """
    llm = LLM_initializer()
    prompt = """
    You are an IELTS Speaking Evaluation Agent.
    Your job:
    - Receive a candidate's speech transcription with time stamps.
    - Evaluate it based on official IELTS Speaking criteria:
    1) Fluency & Coherence
    2) Lexical Resource (Vocabulary)
    3) Grammatical Range & Accuracy
    4) Pronunciation

    Important rules:
    - DO NOT invent facts or evaluation.
    - DO NOT correct the answer unless asked. Your role is evaluator, not teacher.
    - Assume audio → text transcription may come later, but now input is text only.
    - If the user submits an incomplete prompt or unclear text, ask for clarification.

    Tools:
    - You have 4 evaluation tools (fluency, vocabulary, grammar, pronunciation)
    Scoring:
    - Score each category from Band 0 to Band 9
    - Give a final Band score (average of 4 criteria)
    - Follow real IELTS speaking band descriptors (approximate)

    Output format ALWAYS:
    1) Scores table  
    - Fluency:
    - Vocabulary:
    - Grammar:
    - Pronunciation:
    - **Overall Band:**

    2) Justification (bullet points, short and objective)

    Tone guidelines:
    - Be objective, not emotional
    - No slang, no jokes
    - Professional IELTS examiner tone
    """
    agent = create_agent(llm.model, system_prompt=prompt, tools=[fluency_rating, vocabulary_rating, grammatical_rating, pronunciation_rating])
    response = agent.invoke(
        {"messages": [{"role": "user", "content": speech}]}
    )
    return response['messages'][-1].content
if __name__ == "__main__":
    # Sample test speech
    test_speech = """
    Hello, today I want to talk about my hometown. I grow up in a small city near the seaside.
    The weather there is very nice and people are friendly. I like walking on the beach and
    sometimes I go swimming. It is a peaceful place and I hope one day I can live there again.
    """

    print("=== IELTS Agent Test Run ===")
    result = evaluator(test_speech)
    print(result)
