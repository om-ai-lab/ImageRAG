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
- Ensure that the extracted words or phrases are meaningful. Focus on the names of ground targets from a remote sensing perspective. 
- All adjectives should be part of a phrase that includes a target.
- Do not include standalone adjectives such as adjectives that describe size, shape, color, texture, etc. unless they are paired with a target. 
- Do not include keywords or phrases that related to position and orientation.
- Avoid including overly vague words such as 'image', 'picture', 'photo', 'object', etc., unless they are part of a more specific phrase that provides additional context.

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


georsclip_text_template = [
    'a remote sensing image of many {}.',
    'a remote sensing image of a {}.',
    'a remote sensing image of the {}.',
    'a remote sensing image of the hard to see {}.',
    'a remote sensing image of a hard to see {}.',
    'a low resolution remote sensing image of the {}.',
    'a low resolution remote sensing image of a {}.',
    'a bad remote sensing image of the {}.',
    'a bad remote sensing image of a {}.',
    'a cropped remote sensing image of the {}.',
    'a cropped remote sensing image of a {}.',
    'a bright remote sensing image of the {}.',
    'a bright remote sensing image of a {}.',
    'a dark remote sensing image of the {}.',
    'a dark remote sensing image of a {}.',
    'a close-up remote sensing image of the {}.',
    'a close-up remote sensing image of a {}.',
    'a black and white remote sensing image of the {}.',
    'a black and white remote sensing image of a {}.',
    'a jpeg corrupted remote sensing image of the {}.',
    'a jpeg corrupted remote sensing image of a {}.',
    'a blurry remote sensing image of the {}.',
    'a blurry remote sensing image of a {}.',
    'a good remote sensing image of the {}.',
    'a good remote sensing image of a {}.',
    'a remote sensing image of the large {}.',
    'a remote sensing image of a large {}.',
    'a remote sensing image of the nice {}.',
    'a remote sensing image of a nice {}.',
    'a remote sensing image of the small {}.',
    'a remote sensing image of a small {}.',
    'a remote sensing image of the weird {}.',
    'a remote sensing image of a weird {}.',
    'a remote sensing image of the cool {}.',
    'a remote sensing image of a cool {}.',
    'an aerial image of many {}.',
    'an aerial image of a {}.',
    'an aerial image of the {}.',
    'an aerial image of the hard to see {}.',
    'an aerial image of a hard to see {}.',
    'a low resolution aerial image of the {}.',
    'a low resolution aerial image of a {}.',
    'a bad aerial image of the {}.',
    'a bad aerial image of a {}.',
    'a cropped aerial image of the {}.',
    'a cropped aerial image of a {}.',
    'a bright aerial image of the {}.',
    'a bright aerial image of a {}.',
    'a dark aerial image of the {}.',
    'a dark aerial image of a {}.',
    'a close-up aerial image of the {}.',
    'a close-up aerial image of a {}.',
    'a black and white aerial image of the {}.',
    'a black and white aerial image of a {}.',
    'a jpeg corrupted aerial image of the {}.',
    'a jpeg corrupted aerial image of a {}.',
    'a blurry aerial image of the {}.',
    'a blurry aerial image of a {}.',
    'a good aerial image of the {}.',
    'a good aerial image of a {}.',
    'an aerial image of the large {}.',
    'an aerial image of a large {}.',
    'an aerial image of the nice {}.',
    'an aerial image of a nice {}.',
    'an aerial image of the small {}.',
    'an aerial image of a small {}.',
    'an aerial image of the weird {}.',
    'an aerial image of a weird {}.',
    'an aerial image of the cool {}.',
    'an aerial image of a cool {}.',
    'a satellite image of many {}.',
    'a satellite image of a {}.',
    'a satellite image of the {}.',
    'a satellite image of the hard to see {}.',
    'a satellite image of a hard to see {}.',
    'a low resolution satellite image of the {}.',
    'a low resolution satellite image of a {}.',
    'a bad satellite image of the {}.',
    'a bad satellite image of a {}.',
    'a cropped satellite image of the {}.',
    'a cropped satellite image of a {}.',
    'a bright satellite image of the {}.',
    'a bright satellite image of a {}.',
    'a dark satellite image of the {}.',
    'a dark satellite image of a {}.',
    'a close-up satellite image of the {}.',
    'a close-up satellite image of a {}.',
    'a black and white satellite image of the {}.',
    'a black and white satellite image of a {}.',
    'a jpeg corrupted satellite image of the {}.',
    'a jpeg corrupted satellite image of a {}.',
    'a blurry satellite image of the {}.',
    'a blurry satellite image of a {}.',
    'a good satellite image of the {}.',
    'a good satellite image of a {}.',
    'a satellite image of the large {}.',
    'a satellite image of a large {}.',
    'a satellite image of the nice {}.',
    'a satellite image of a nice {}.',
    'a satellite image of the small {}.',
    'a satellite image of a small {}.',
    'a satellite image of the weird {}.',
    'a satellite image of a weird {}.',
    'a satellite image of the cool {}.',
    'a satellite image of a cool {}.',
]



clip_text_template = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

paraphrase_template = naive_paraphrase_template
keyword_template = naive_keyword_template
text_expansion_template = naive_text_expansion_template