GRAPH_PROMPT = """
Please analyze the given video and extract key information in a structured JSON format in English. Identify and describe:

Entities: List all distinct objects, people, animals, or other significant elements present in the video.
Actions: If the entities are interacting, describe their actions and relationships in a structured manner.
Scenes: Identify and describe the locations, environments, or contexts where the events occur.
If the video is filmed from a first-person point of view, please also describe "subject" as "me" and actions or interactions from this person.
Ensure the output strictly follows the JSON format below:
{
    "entities": [{"entity name": "", "description": ""}],
    "actions": [{"entity name": "", "action description": ""}],
    "scenes": [{"location": ""}]
}
The "entity name" in actions should belong to "entity name" in entities.
Each section should be detailed but concise, capturing all relevant interactions and contextual elements from the video. Avoid unnecessary text outside the JSON output.
"""

REASONING_PROMPT = """Given a question of a long video and potential candidates: 
Question: {query}

Candidates: {candidates}

You need to retrieve the relevant video segments to answer the question. Note that you do not need to see the video. But based on the question please think step by step what are the important things for retrieval.

[keywords] Please identify the information, like entities, scene, action from the question that is important to retrieve the segments for further answer the question. Do not include the candidates in the keywords.

[candidates_necessary] Do you think the information in the candidates is necessary for retrieval? Answer yes or no.

[multiple] Do you think it needs to aggregate the information from multiple segments to answer the question? ONLY answer yes or no.

[time] Please identify if it can tell the question is asking which part of the video. Answer begin, end or none.

[tool] Do you think it needs additional step for answering the question, please select from [object counting, action counting, order, none].

[global] Can this question be answered based on the overall understanding of the whole video? (e.g., “What is the main topic of the video?” or “What is the main content of the video?”)

Please output the final answer in json format, for example:
{{"multiple": "no", "keywords": ["man in black"], "time": "begin", "tool": 'none', 'candidates_necessary': 'yes', 'global': 'yes'}}
"""

PRED_PROMPT = "Respond with only the letter of the correct option.\n"

PRED_SQL_PROMPT = "Given the useful information: {sql_input}. But be aware that it may still omit crucial details so it's essential to check the video for completeness.\n"

SQL_PROMPT = """Given a question of a long video and potential candidates: 
Question: {query}

Candidates: {candidates}

Given a multiple-choice question about a video, break it down into several sub-questions that analyze the key elements required to answer it step by step.

Form yes/no or counting questions to verify the presence of the subject or event in the video (e.g., "Does the video show [subject/event]?").
Ensure the sub-questions cover all necessary aspects to reach the correct answer.

Example 1:
Question: Which of the following statements is not correct?
Candidates: 
A. The Titanic finally sank because 5 adjacent compartments were breached.
B. Despite the lack of lifeboats, the Titanic met all the requirement.
C. People on the Titanic were not rescued in time because its operator was sleeping.
D. The Titanic was equipped with 20 lifeboats.
Your output:
{{
    "Q1": "Does the video show the Titanic finally sank because 5 adjacent compartments were breached?"
    "Q2": "Does the video show the Titanic met all the requirement despite the lack of lifeboats?"
    "Q3": "Does the video show people on the Titanic were not rescued in time because its operator was sleeping?"
    "Q4": "Does the video show the Titanic was equipped with 20 lifeboats?"
}}
Example 2:
Question: How many ships are shown in the map while the sinking ship sending out message?
Candidates: A. 3. B. 8. C. 11. D. 9.
Your output:
{{
    "Q1": "Is there a map showing a sinking ship sending out a message?"
}}
Example 3:
Question: What is the score at the end of the half?
Candidates: A. 38 - 31. B. 38 - 34. C. 67 - 61. D. 67 - 60.
Your output:
{{
    "Q1": "Is the video showing the game at the end of the half?"
}}
Example 4:
Question: Which athlete in the video was the first to touch off the crossbar?
Candidates: A. Athlete from Russia. B. Athlete from Qatar. C. Athlete from Canada. D. Athlete from Ukraine.
Your output:
{{
    "Q1": "Is the athlete from Russia touch off the crossbar?"
    "Q2": "Is the athlete from Qatar touch off the crossbar?"
    "Q3": "Is the athlete from Canada touch off the crossbar?"
    "Q4": "Is the athlete from Ukraine touch off the crossbar?"
}}
Example 5:
Question: According to the video, what is the chronological order in which the following actions occur?\n(a) Weaving in the ends.\n(b) Crocheting a single crochet.\n(c) Finishing the handcraft.\n(d) Making a slip knot.\n(e) Crocheting a chain.
Your output:
{{
    "Q1": "Is there a scene showing weaving in the ends?"
    "Q2": "Is there a scene showing crocheting a single crochet?"
    "Q3": "Is there a scene showing finishing the handcraft?"
    "Q4": "Is there a scene showing making a slip knot?"
    "Q5": "Is there a scene showing crocheting a chain?"
}}
"""

SQL_ANSWER_COUNT_PROMPT = """You are a video segment analyzer. You will be given a counting question about a video.
Your task is to provide a numeric answer only (no text or explanation) in JSON format.
If the number of the specified object or event cannot be determined from the video, you MUST respond with 0.
Questions: {questions}

Questions: {{
"Q1": "How many ..."
}}
Your output:
{{
    "Q1": 2
}}
"""

SQL_ANSWER_PROMPT = """Given a list of questions related to the video, generate corresponding answers in JSON format.
The answer must be either "yes" or "no".
Do not provide any additional explanations or responses beyond the required format.
Questions: {questions}

For Example:
Questions: {{
"Q1": "Is there ...",
"Q2": "Does the video show ..."
}}
Your output
{{
    "Q1": "yes",
    "Q2": "no"
}}

Ensure that each response adheres strictly to the specified answer types.\n
"""

AGGREGATE_PROMPT = """You are given a multiple-choice question and a set of sub-questions with their corresponding answers for different video segments.
Your task is to aggregate the information over the video segments and cancel out options from the candidates that are contradicted by the information.
The numbers associated with video segments indicate their order in time, arranged in ascending temporal order. For example, video0 happens before video1.
Question: {query}
Candidates: {candidates}
Information: {input}

Summarize the information gathered from the video segments within 20 words. Avoid giving or ruling out final answers, since some details may be missing.
"""

CAPTION_PROMPT = """You are given a question, a set of answer options, and the captions of a video.
Your task is to identify which caption segments are relevant to the question.
Return a list of time indices of captions that are related to the question. If no relevant captions are found, return an empty list.
Question: {query}
Options: {candidates}
Captions: {captions}

For example:
Your output: [5, 20]
Limit the number of time indices to at most 10.
"""

CAPTION_GENERATION_PROMPT = """You are provided with a video clip.
Your task is to generate a concise and descriptive caption for this clip.
The caption should summarize the main action or event occurring in the video.
Focus on the visual content and avoiding hallucinating details not present in the video.
"""


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------

import re as _re

# Tags that models use for chain-of-thought / thinking output.
# Each tuple is (open_tag_pattern, close_tag_pattern).
_THINKING_TAG_PAIRS = [
    (r"<analysis>", r"</analysis>"),
    (r"<think>", r"</think>"),
    (r"<reasoning>", r"</reasoning>"),
    (r"<thought>", r"</thought>"),
]


def strip_thinking_tags(text: str) -> str:
    """Remove model thinking/analysis tags and their content from *text*.

    Many VLMs (e.g. KeyeVL, Qwen3) wrap their response in
    ``<analysis>...</analysis>`` or ``<think>...</think>`` blocks.  This
    function strips those blocks and returns only the actual answer.

    If stripping leaves nothing, the original text is returned.
    """
    if not text:
        return text
    cleaned = text
    for open_pat, close_pat in _THINKING_TAG_PAIRS:
        cleaned = _re.sub(
            open_pat + r".*?" + close_pat,
            "",
            cleaned,
            flags=_re.DOTALL | _re.IGNORECASE,
        )
    cleaned = cleaned.strip()
    return cleaned if cleaned else text

