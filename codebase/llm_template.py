naive_paraphrase_template = """
Task: Rewrite a query with complex references into a clearer query.

Guidelines: 
- Simplify the language and remove unnecessary assumptions or references.
- Make sure the rewritten query is clear and easy to understand.
- Retain the original meaning of the query.

Example:
Input query: "Suppose the top of this image represents north. How many aircraft are heading northeast? What is the color of the building rooftop to their southeast?"
Output: "How many aircraft are heading in the up-right direction? What is the color of the building rooftop below them?"

Your task is to rewrite the following complex query into a simpler and clearer one:

Input: {}
Output: 
"""

naive_keyword_template = """
Task: Extract keywords or key phrases from a given query sentence.

Guidelines: 
- Analyze the sentence and identify the important keywords or phrases. These words or phrases should represent the core content or main information of the sentence. 
- Ensure that the extracted words or phrases are meaningful and can be used to quickly understand the main idea of the sentence.
- Focus on the names of ground targets in the view of remote sensing. Include the phrases that describe size, shape, color, texture, and spatial relation of them.

Example:
Input query:  "How many aircraft are heading in the up-right direction? What is the color of the building rooftop below them?"
Output: ["aircraft", "building rooftop", "aircraft are heading in the up-right direction", "color of the building rooftop"]

Your task is to apply the same process to the following sentence:

Input query: {}
Output: 
"""

naive_text_expansion_template =  """
Task: Write a few sentences (output in string) to explain the phrase I give you.

Guidelines: 
- Provide a detailed explanation including synonyms (if they exist) of the phrase. 
- The explanation should be comprehensive enough to match the phrase with relevant class names based on sentence embedding similarity.

Example:
Input phrase:  "Airport"
Output: "An airport is a facility where aircraft, such as airplanes and helicopters, take off, land, and are serviced. It typically consists of runways, taxiways, terminals for passenger and cargo handling, control towers for air traffic management, and hangars for aircraft maintenance. Airports serve as hubs for commercial aviation, connecting travelers and goods between local and international destinations. Synonyms for 'airport' include airfield, aerodrome, airstrip, and terminal (in certain contexts, referring specifically to the passenger facilities."

Your task is to apply the same process to the phrase provided:

Input query: {}
Output: 
"""

paraphrase_template = naive_paraphrase_template
keyword_template = naive_keyword_template
text_expansion_template = naive_text_expansion_template