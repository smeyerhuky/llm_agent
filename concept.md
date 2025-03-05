I need helping generating 'instructions' for a custom GPT.

This palceholder text shows an example:
```example
Think step by step and show reasoning for complex problems. Use specific examples.

Break down large tasks and ask clarifying questions when needed.

Use Artifacts only for web apps and code demos.

When giving feedback, explain thought process and highlight issues and opportunities.
```

The agent's instructions require:
- < 5k words
- Very detail oriented with background, context, few shot examples, etc..
- A description of the formatting required of all conversations
- An understanding of how to communicate large-scale systems and architectural designs
  - Able to do this with uml (such as plantuml)
  - Able to understand how to implement a given design skeleton as a docker-compose project or swarm.
- Able to communicate with product and design.
  - Has several years of design experience as well so can describe images to create fonts, svg, or other iconofgraphy.
- Has over 25 years of experience in the technology sector. This assistant has helped build massive projects at several institutions, research, university, and big tech (FAANG)
- Has worked on the kindle team optimzing device characteristics using neural networks to study battery usage and screen reading limitations
- Has worked on large language models and implemented agentic AI at several small and large scale systems
- Worked in the c6 (iowa state virtual reality lab) implemnting user-feedback low-latency models of learned behavior for parallax visual queues in a 3 dimensional space.
  - this was used by boeing, nasa, etc to geneate fly throughs of space and air sectors with realistic movement detection.
- I've worked on large scale learned behavior systems and have built low level neural networks that can help auto-classify larger models.
- Worked in many experiments with computer vision as well as 3d structured light and machine vision
- Has experience paralellizing and breaking down large scale autuomation concepts and turning them in to safe and highly scalable distributed systems.
    - This distributed systems experience is not limited to internet and cloud systems. It is also built on top of OS and low-level cpu optimizations done via assembly. The main concept is to learn to use a parallelization and multiple processing units (CPU and GPU) to build a highly distributed system.
    - Building on techniques from low-level kernel and OS designs the assistant should be capable of adapting these abstractions to distributing the llm_agent architecture I am designing as highly distributed and agentic for a local model.

- The assistant is always trained to triage it's resposes with a moderation, censor, and verification  layer that ensures output is always consistent (which  might mean hard coding results and caching them as a part of a model or manifest to use later.)


Here is an example:
I'll clean up the image outputs in the conversations. Here's the updated version:

```
I have been working in software and hardware automation for years. One major abstraction I've learned is to break up large goals in to smaller more easily acheivable goals. This allows me to break down regular repetitive steps in a process and learn to identify which is the best to build and where is it a good idea to look for an off the shelf solution.
Over the many contributions I've made I've learned that building a small set of patterns, scripts, processes, and tools to be a part of a larger tool chain is a great way to keep your learnings and take them with you. For my agent_llm project I need an assistant that will be my partner in identifying a way to create and implement common and stadard automation orchestration but some of the layers are powered by an AI agent. The agent is intended to help stitch together the parts of that system and answer the queries being asked of it. For example, this agent might get asked to do a simple thing like create an SVG of a smiley face. To do this. The Agent will do the following:
- Pass the query to a quick moderation layer to determine if a simple response from an llm is ok to answer or if it needs a more detailed 'code' execution instead.
- This moderation layer has its own agent and safeguards that will have been put together by experts that have automation expertise in this field.
- After the moderation layer determines: [answer, code_needed] it will go down the path of answering the question or generating code to collect what it needs to actually answer the question.
- Locally this will mean breaking up the request in to a set of tasks that follow a predetermined template (to be designed).
- This template will have systemic requirements as a checklist and generate several llm responses using prompts.py and other tempaltes specially designed to extract the most contextual result and format it so the API can use json to parse the results and get {response, code, [requirements], simple_name_identifier, step_id, trace_id, span_id, metrics...}
- There are too many layers to iterate on, but the main point is that any 'agent' is essentially an llm-backed prompt customized to be contextual and just a 'one small part in a larger system'.
- All results from the agent will eventually be broken down into a manifest. This manifest is searched in the moderation layer before each prompt is initially handled. This allows the layer to break down the task and then look up any parts of the solution that are in the manifest.
- This allows for all the layers to be 'memory' based.
Here is a sample interaction:
"User: Get the most common word on the wiki page for Chabad."
**---**
"llm_agent: Sure, I will collect the Chabad wiki page and determine the most common words."
"llm_agent: Thinking..."
"llm_agent: To do this task, I will need to run the following pipeline of actions: "
"llm_agent: [ Fetch Wikipedia Page ] -> [ Parse To Text ] -> [ Create Frequency Distribution ] -> [ return top word and count ]"
"llm_agent: -----
"llm_agent: I have taken some liberties and some assumptions in my pipeline above. Would you like me to execute this?"
**---**
"User: Yes Please"
"llm_agent: Breaking down this work..."
"llm_agent: I will create or use the following modules:"
"llm_agent: - CREATE_MODULE(wiki_helper)"
"llm_agent: - Prompt: Create a wiki request l
"llm_agent: - ADD_MODULE("fetch")
"llm_agent: - Prompt: "Generate a module to fetch wiki documents using (url=default(urls.wikipedia))
User: Can you generate an SVG of a bar chart showing the population growth of the top 5 fastest growing US cities?
llm_agent: I'll help you create an SVG bar chart showing population growth for the top 5 fastest growing US cities.
llm_agent: Thinking...
llm_agent: To complete this task, I need to execute the following pipeline:
[ Gather Population Data ] -> [ Process Growth Rates ] -> [ Select Top 5 ] -> [ Generate SVG Bar Chart ]
llm_agent: I've made some assumptions about the timeframe for growth (using recent years). Would you like me to proceed with this approach?
User: Yes, please proceed.
llm_agent: Breaking down this work...
llm_agent: I will create or use the following modules:
- CREATE_MODULE(data_collector)
- ADD_MODULE("census_api")
- ADD_MODULE("data_processor")
- CREATE_MODULE(svg_generator)
llm_agent: Creating python modules...
llm_agent: Orchestrating modules...
llm_agent: Executing data collection...
llm_agent: I've gathered population data for US cities between 2010-2020 and calculated growth rates.
llm_agent: Orchestrating results
llm_agent: The top 5 fastest growing cities are:
1. Frisco, TX (71.1%)
2. Buckeye, AZ (68.4%)
3. New Braunfels, TX (56.7%)
4. McKinney, TX (51.9%)
5. South Jordan, UT (51.8%)
llm_agent: Generating SVG visualization...
llm_agent: Here's your SVG bar chart:

<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
 <style>
 .bar { fill: #4285f4; }
 .bar:hover { fill: #3367d6; }
 text { font-family: Arial; }
 .title { font-size: 24px; text-anchor: middle; }
 .label { font-size: 12px; text-anchor: middle; }
 .value { font-size: 14px; text-anchor: middle; fill: white; }
 </style>
 <text x="300" y="30" class="title">Population Growth (2010-2020)</text>
 <!-- Bars -->
 <rect x="50" y="70" width="80" height="280" class="bar" />
 <rect x="150" y="90" width="80" height="260" class="bar" />
 <rect x="250" y="145" width="80" height="205" class="bar" />
 <rect x="350" y="170" width="80" height="180" class="bar" />
 <rect x="450" y="172" width="80" height="178" class="bar" />
 <!-- City Labels -->
 <text x="90" y="370" class="label">Frisco, TX</text>
 <text x="190" y="370" class="label">Buckeye, AZ</text>
 <text x="290" y="370" class="label">New Braunfels, TX</text>
 <text x="390" y="370" class="label">McKinney, TX</text>
 <text x="490" y="370" class="label">South Jordan, UT</text>
 <!-- Values -->
 <text x="90" y="210" class="value">71.1%</text>
 <text x="190" y="220" class="value">68.4%</text>
 <text x="290" y="250" class="value">56.7%</text>
 <text x="390" y="270" class="value">51.9%</text>
 <text x="490" y="270" class="value">51.8%</text>
</svg>

...
...
**---**

Conversation 2:
```
User: Present an image of Abraham Lincoln's face

llm_agent: I'll help you present an image of Abraham Lincoln's face.

llm_agent: Thinking...

llm_agent: To do this task, I will need to run the following pipeline of actions:
[ Fetch Wikipedia Page ] -> [ Extract Images ] -> [ Filter for Face Images ] -> [ Verify Face Detection ] -> [ Present Image ]

llm_agent: I'll need to search for Abraham Lincoln's Wikipedia page, extract images, identify those containing his face, verify with face detection, and present the best result. Would you like me to proceed with this approach?

User: Yes, please go ahead

llm_agent: Breaking down this work...

llm_agent: I will create or use the following modules:
- USE_MODULE("wiki_helper.fetch")
- CREATE_MODULE(image_extractor)
- USE_MODULE("image_processor.face_detection")
- CREATE_MODULE(presentation_formatter)

llm_agent: Executing modules...

llm_agent: Fetching Wikipedia data for "Abraham Lincoln"...
llm_agent: Successfully retrieved Wikipedia page with 47 images.
llm_agent: Filtering images for portraits of Abraham Lincoln...
llm_agent: Running face detection on candidate images...
llm_agent: Found 8 images containing clear faces, selecting the most prominent official portrait.
llm_agent: Processing image for optimal display...

llm_agent: Here is the image of Abraham Lincoln's face:

<svg width="400" height="500" xmlns="http://www.w3.org/2000/svg">
  <image href="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCADIAKADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL..." x="0" y="0" width="400" height="500"/>
  <text x="200" y="480" font-family="Arial" font-size="14" text-anchor="middle">Abraham Lincoln, 16th U.S. President (1861-1865)</text>
</svg>

llm_agent: I've presented an official portrait of Abraham Lincoln from his Wikipedia page. This image shows his distinctive facial features including his beard and serious expression.
```

Conversation 3:
```
User: Compare the faces of Abraham Lincoln and George Washington

llm_agent: I'll help you compare the faces of Abraham Lincoln and George Washington.

llm_agent: Thinking...

llm_agent: To do this task, I will need to run the following pipeline of actions:
[ Fetch Both Wikipedia Pages ] -> [ Extract Face Images ] -> [ Process Images for Comparison ] -> [ Identify Facial Features ] -> [ Generate Visual Comparison ] -> [ Present Analysis ]

llm_agent: I'll gather facial images of both presidents, extract key facial features, and create a side-by-side comparison with annotations highlighting the differences. Would you like me to proceed with this approach?

User: Yes, that sounds great

llm_agent: Breaking down this work...

llm_agent: I will create or use the following modules:
- USE_MODULE("wiki_helper.fetch")
- USE_MODULE("image_extractor")
- USE_MODULE("image_processor.face_detection")
- CREATE_MODULE(facial_feature_analyzer)
- CREATE_MODULE(comparison_visualizer)

llm_agent: Executing modules...

llm_agent: Fetching Wikipedia data for "Abraham Lincoln" and "George Washington"...
llm_agent: Successfully retrieved both Wikipedia pages.
llm_agent: Extracting portrait images using existing image_extractor module...
llm_agent: Running face detection on candidate images...
llm_agent: Selected primary portrait images for both presidents.
llm_agent: Analyzing facial features...
llm_agent: Creating side-by-side comparison with feature annotations...

llm_agent: Here is the visual comparison of Abraham Lincoln and George Washington's faces:

<svg width="800" height="500" xmlns="http://www.w3.org/2000/svg">
  <!-- Washington Portrait -->
  <image href="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a..." x="50" y="50" width="300" height="375"/>
  
  <!-- Lincoln Portrait -->
  <image href="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4n..." x="450" y="50" width="300" height="375"/>
  
  <!-- Labels -->
  <text x="200" y="450" font-family="Arial" font-size="16" text-anchor="middle">George Washington (1789-1797)</text>
  <text x="600" y="450" font-family="Arial" font-size="16" text-anchor="middle">Abraham Lincoln (1861-1865)</text>
  
  <!-- Comparison Lines & Annotations -->
  <line x1="200" y1="150" x2="600" y2="170" stroke="red" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="400" y="140" font-family="Arial" font-size="14" text-anchor="middle" fill="red">Hairline Difference</text>
  
  <line x1="200" y1="200" x2="600" y2="220" stroke="blue" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="400" y="190" font-family="Arial" font-size="14" text-anchor="middle" fill="blue">Facial Hair</text>
  
  <line x1="180" y1="250" x2="580" y2="250" stroke="green" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="400" y="240" font-family="Arial" font-size="14" text-anchor="middle" fill="green">Nose Structure</text>
  
  <line x1="200" y1="300" x2="600" y2="300" stroke="purple" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="400" y="290" font-family="Arial" font-size="14" text-anchor="middle" fill="purple">Jaw Shape</text>
</svg>

llm_agent: Based on my analysis of their facial features:

1. Facial Hair: Washington is clean-shaven while Lincoln has his iconic beard and prominent sideburns
2. Face Shape: Washington has a rounder face while Lincoln's is longer and more angular
3. Nose: Washington's nose appears broader while Lincoln's is more prominent
4. Expression: Washington typically appears more stoic in portraits while Lincoln's expressions often convey melancholy
5. Hair: Washington wore powdered wigs in the 18th-century style while Lincoln had his natural dark hair

The visualization highlights these key differences between the two historic presidents.
```

Conversation 4:
```
User: Create an interactive timeline showing the major inventions of Thomas Edison with images

llm_agent: I'll help you create an interactive timeline of Thomas Edison's major inventions with images.

llm_agent: Thinking...

llm_agent: To do this task, I will need to run the following pipeline of actions:
[ Research Edison's Inventions ] -> [ Collect Chronological Data ] -> [ Gather Invention Images ] -> [ Design Interactive Timeline ] -> [ Implement User Interaction ] -> [ Present Final Timeline ]

llm_agent: I'll research Edison's key inventions chronologically, collect appropriate images, and create an interactive SVG timeline with invention details appearing on hover/click. Would you like me to proceed with this approach?

User: Yes, please proceed

llm_agent: Breaking down this work...

llm_agent: I will create or use the following modules:
- USE_MODULE("wiki_helper.fetch")
- USE_MODULE("image_extractor")
- CREATE_MODULE(timeline_data_processor)
- CREATE_MODULE(interactive_timeline_generator)
- CREATE_MODULE(svg_interaction_handler)

llm_agent: Executing modules...

llm_agent: Fetching data about Thomas Edison and his inventions...
llm_agent: Collecting and organizing chronological invention data...
llm_agent: Identifying 8 major inventions for timeline inclusion...
llm_agent: Gathering images for each invention...
llm_agent: Designing interactive timeline layout...
llm_agent: Implementing hover/click interaction functionality...

llm_agent: Here is the interactive timeline of Thomas Edison's major inventions:

<svg width="900" height="500" xmlns="http://www.w3.org/2000/svg">
  <style>
    .timeline-line { stroke: #ccc; stroke-width: 4; }
    .time-point { fill: #4285f4; stroke: #2952a3; stroke-width: 2; r: 8; cursor: pointer; }
    .time-point:hover { r: 10; fill: #2952a3; }
    .time-label { font-family: Arial; font-size: 12px; text-anchor: middle; }
    .invention-image { opacity: 0; transition: opacity 0.3s; }
    .invention-details { font-family: Arial; font-size: 14px; opacity: 0; transition: opacity 0.3s; }
    .year-marker { font-family: Arial; font-size: 10px; fill: #666; text-anchor: middle; }
  </style>
  
  <!-- Timeline base -->
  <line x1="100" y1="250" x2="800" y2="250" class="timeline-line" />
  
  <!-- Year markers -->
  <text x="100" y="270" class="year-marker">1870</text>
  <text x="250" y="270" class="year-marker">1880</text>
  <text x="400" y="270" class="year-marker">1890</text>
  <text x="550" y="270" class="year-marker">1900</text>
  <text x="700" y="270" class="year-marker">1910</text>
  
  <!-- Invention 1: Stock Ticker (1869) -->
  <circle cx="90" cy="250" class="time-point" onclick="showInvention(1)" />
  <text x="90" y="290" class="time-label">Stock Ticker</text>
  <text x="90" y="305" class="time-label">1869</text>
  <image id="img1" x="50" y="50" width="150" height="150" class="invention-image" 
         href="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAg..." />
  <text id="details1" x="450" y="420" class="invention-details" text-anchor="middle">
    Stock Ticker (1869): Edison's first patented invention, an improved stock ticker
    that synchronized multiple tickers and reduced transmission errors.
  </text>
  
  <!-- Invention 2: Phonograph (1877) -->
  <circle cx="230" cy="250" class="time-point" onclick="showInvention(2)" />
  <text x="230" y="290" class="time-label">Phonograph</text>
  <text x="230" y="305" class="time-label">1877</text>
  <image id="img2" x="50" y="50" width="150" height="150" class="invention-image"
         href="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAg..." />
  <text id="details2" x="450" y="420" class="invention-details" text-anchor="middle">
    Phonograph (1877): Edison's favorite invention, the first device that could 
    both record sound and play it back, using tinfoil-wrapped cylinders.
  </text>
  
  <!-- Additional inventions would be included similarly -->
  
  <!-- Simple JavaScript for interaction -->
  <script type="text/javascript">
    function showInvention(id) {
      // Hide all images and details
      for (let i = 1; i <= 8; i++) {
        document.getElementById('img' + i).style.opacity = 0;
        document.getElementById('details' + i).style.opacity = 0;
      }
      // Show selected invention
      document.getElementById('img' + id).style.opacity = 1;
      document.getElementById('details' + id).style.opacity = 1;
    }
  </script>
</svg>

llm_agent: I've created an interactive timeline showing Thomas Edison's major inventions from 1869 to 1914. The timeline includes the following inventions:

1. Stock Ticker (1869)
2. Phonograph (1877)
3. Carbon Microphone (1877)
4. Electric Light Bulb (1879)
5. Electric Power Distribution (1882)
6. Motion Picture Camera (1891)
7. Fluoroscope (1896)
8. Nickel-Iron Battery (1901)

Click on any point in the timeline to see an image of the invention and read details about it. The timeline visualizes Edison's innovative contributions spanning over four decades.
```

Conversation 5:
```
User: Analyze the sentiment trends in news articles about artificial intelligence from the past month and visualize the results

llm_agent: I'll analyze sentiment trends in recent AI news articles and create a visualization of the results.

llm_agent: Thinking...

llm_agent: To do this task, I will need to run the following pipeline of actions:
[ Collect Recent AI News Articles ] -> [ Extract Text Content ] -> [ Perform Sentiment Analysis ] -> [ Identify Key Topics ] -> [ Process Temporal Trends ] -> [ Generate Data Visualization ] -> [ Present Analysis ]

llm_agent: I'll gather recent AI news, analyze sentiment scores over time, identify key topic clusters, and create visualizations showing how sentiment around different AI topics has evolved. Would you like me to proceed with this approach?

User: Yes, please go ahead

llm_agent: Breaking down this work...

llm_agent: I will create or use the following modules:
- CREATE_MODULE(news_collector)
- CREATE_MODULE(article_processor)
- CREATE_MODULE(sentiment_analyzer)
- CREATE_MODULE(topic_classifier)
- CREATE_MODULE(trend_analyzer)
- USE_MODULE("visualization_generator")

llm_agent: Executing modules...

llm_agent: Collecting recent AI news articles from multiple sources...
llm_agent: Retrieved 237 articles from the past month.
llm_agent: Extracting and cleaning text content...
llm_agent: Performing sentiment analysis on all articles...
llm_agent: Classifying articles into topic clusters...
llm_agent: Identified 5 major topic clusters: AI Regulation, Generative AI, AI Ethics, AI in Healthcare, AI in Business
llm_agent: Analyzing temporal sentiment trends for each topic...
llm_agent: Generating visualization of sentiment trends...

llm_agent: Here is the visualization of AI news sentiment trends:

<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <style>
    .grid-line { stroke: #e0e0e0; stroke-width: 1; }
    .axis-label { font-family: Arial; font-size: 12px; }
    .title { font-family: Arial; font-size: 20px; text-anchor: middle; }
    .subtitle { font-family: Arial; font-size: 14px; text-anchor: middle; }
    .line { fill: none; stroke-width: 3; }
    .legend-item { font-family: Arial; font-size: 12px; }
    .annotation { font-family: Arial; font-size: 10px; text-anchor: start; }
  </style>
  
  <!-- Title and subtitle -->
  <text x="400" y="30" class="title">Sentiment Trends in AI News (Past Month)</text>
  <text x="400" y="55" class="subtitle">Daily average sentiment by topic (Feb 4 - Mar 4, 2025)</text>
  
  <!-- Y-axis -->
  <line x1="60" y1="100" x2="60" y2="500" stroke="black" stroke-width="2" />
  <text x="55" y="100" class="axis-label" text-anchor="end">+1.0</text>
  <text x="55" y="200" class="axis-label" text-anchor="end">+0.5</text>
  <text x="55" y="300" class="axis-label" text-anchor="end">0.0</text>
  <text x="55" y="400" class="axis-label" text-anchor="end">-0.5</text>
  <text x="55" y="500" class="axis-label" text-anchor="end">-1.0</text>
  <text x="30" y="300" class="axis-label" text-anchor="middle" transform="rotate(-90, 30, 300)">Sentiment Score</text>
  
  <!-- X-axis -->
  <line x1="60" y1="500" x2="740" y2="500" stroke="black" stroke-width="2" />
  <text x="100" y="520" class="axis-label" text-anchor="middle">Feb 4</text>
  <text x="240" y="520" class="axis-label" text-anchor="middle">Feb 11</text>
  <text x="380" y="520" class="axis-label" text-anchor="middle">Feb 18</text>
  <text x="520" y="520" class="axis-label" text-anchor="middle">Feb 25</text>
  <text x="660" y="520" class="axis-label" text-anchor="middle">Mar 4</text>
  <text x="400" y="545" class="axis-label" text-anchor="middle">Date (2025)</text>
  
  <!-- Grid lines -->
  <line x1="60" y1="100" x2="740" y2="100" class="grid-line" />
  <line x1="60" y1="200" x2="740" y2="200" class="grid-line" />
  <line x1="60" y1="300" x2="740" y2="300" class="grid-line" />
  <line x1="60" y1="400" x2="740" y2="400" class="grid-line" />
  
  <!-- Data lines -->
  <path d="M60,290 L100,270 L140,250 L180,220 L220,210 L260,240 L300,260 L340,280 L380,250 L420,230 L460,210 L500,190 L540,180 L580,210 L620,240 L660,250 L700,270 L740,260" class="line" stroke="#4285F4" />
  <path d="M60,310 L100,330 L140,350 L180,370 L220,320 L260,300 L300,290 L340,280 L380,300 L420,310 L460,290 L500,270 L540,250 L580,230 L620,220 L660,230 L700,250 L740,240" class="line" stroke="#DB4437" />
  <path d="M60,330 L100,340 L140,370 L180,380 L220,400 L260,410 L300,420 L340,430 L380,410 L420,370 L460,340 L500,330 L540,320 L580,310 L620,300 L660,310 L700,330 L740,350" class="line" stroke="#F4B400" />
  <path d="M60,200 L100,210 L140,200 L180,190 L220,180 L260,170 L300,160 L340,150 L380,170 L420,190 L460,210 L500,200 L540,210 L580,220 L620,210 L660,200 L700,190 L740,180" class="line" stroke="#0F9D58" />
  <path d="M60,270 L100,280 L140,290 L180,270 L220,250 L260,240 L300,230 L340,240 L380,260 L420,270 L460,280 L500,290 L540,280 L580,270 L620,260 L660,280 L700,290 L740,300" class="line" stroke="#9C27B0" />
  
  <!-- Legend -->
  <rect x="580" y="100" width="12" height="12" fill="#4285F4" />
  <text x="600" y="110" class="legend-item">AI Regulation</text>
  <rect x="580" y="125" width="12" height="12" fill="#DB4437" />
  <text x="600" y="135" class="legend-item">Generative AI</text>
  <rect x="580" y="150" width="12" height="12" fill="#F4B400" />
  <text x="600" y="160" class="legend-item">AI Ethics</text>
  <rect x="580" y="175" width="12" height="12" fill="#0F9D58" />
  <text x="600" y="185" class="legend-item">AI in Healthcare</text>
  <rect x="580" y="200" width="12" height="12" fill="#9C27B0" />
  <text x="600" y="210" class="legend-item">AI in Business</text>
  
  <!-- Annotations for key events -->
  <circle cx="220" cy="210" r="5" fill="red" />
  <text x="225" y="190" class="annotation">Major EU AI Act announcement</text>
  <line x1="220" y1="210" x2="225" y2="190" stroke="red" stroke-width="1" />
  
  <circle cx="460" cy="210" r="5" fill="red" />
  <text x="465" y="190" class="annotation">New GPT-5 release</text>
  <line x1="460" y1="210" x2="465" y2="190" stroke="red" stroke-width="1" />
  
  <circle cx="580" cy="230" r="5" fill="red" />
  <text x="585" y="210" class="annotation">Apple AI product launch</text>
  <line x1="580" y1="230" x2="585" y2="210" stroke="red" stroke-width="1" />
</svg>

llm_agent: Based on my analysis of 237 AI news articles from the past month (February 4 - March 4, 2025), I've identified several notable sentiment trends:

1. Healthcare AI coverage remains consistently positive (+0.5 to +0.8 sentiment score), with advancements in diagnostic tools receiving particularly favorable coverage.

2. AI Regulation sentiment improved from neutral to slightly positive following the EU AI Act clarifications in mid-February.

3. Generative AI sentiment started negative but has been trending positive since late February, coinciding with the GPT-5 release featuring improved accuracy claims.

4. AI Ethics shows the most negative sentiment overall, though it improved slightly in late February.

5. Business applications of AI maintain relatively neutral sentiment with minor fluctuations.

The visualization tracks daily average sentiment


Here is the draft project layout is described below:
# Project Layout

Below is a sample directory structure to illustrate how you can split your agent into multiple files. You can freely rename or reorganize them to suit your needs:

```
my_llm_agent/
├── app.py
├── docker_executor.py
├── faiss_cache.py
├── prompts.py
├── run_code.py
├── session_manager.py
├── modules/
│   └── (Generated reusable modules go here)
└── ...
```

1. **`app.py`**: Contains your main entrypoint with the `main_loop()`.  
2. **`session_manager.py`**: Manages session-wide data, including the global `session_context`.  
3. **`faiss_cache.py`**: Contains all logic for DB initialization, storing prompts in the cache, FAISS indexing, and looking up existing caches.  
4. **`docker_executor.py`**: Handles Docker Compose files, container orchestration, and related operations.  
5. **`run_code.py`**: Ties together Docker execution (using `docker_executor.py`), plus any refinement attempts.  
6. **`prompts.py`**: Contains code that interacts with OpenAI for classification, summarization, refinements, final pass JSON generation, etc.  
7. **`requirements.txt`**: Contains package dependencies.
8. **`db_cleaner.py`**: Contains the cleanup and simple interactions for the init_db functionalitry.
In this example, we assume you still have your `docker-compose-executor.yml` somewhere that you read and write to. The only difference is that the logic for reading/writing that file is now neatly inside `docker_executor.py`.

Below is **a complete example** that you can adapt directly. Where possible, we have:

- Preserved your original docstrings (or consolidated them).
- Maintained the same variable names (e.g., `FAST_MODEL`, `ADVANCED_MODEL`, etc.).
- Ensured your flow of classification → caching → Docker execution → refinement → final pass remains intact.

### Usage / Execution

1. **Install dependencies** (e.g. `pip install -r requirements.txt`).  
2. **Run the app** with `python app.py`.
3. You can still do:
   - `exit` or `quit` to stop,
   - `show context` to see session data,
   - `scrape: <URL>` to fetch content,
   - etc.  
4. The code flow remains:  
   - Check the FAISS / code cache  
   - Possibly re-run or ignore  
   - If no match, classify the request -> code or no_code  
   - Generate code (or text)  
   - Summarize, judge, feedback, final pass  
   - Optionally store module.  

And importantly, the **flow** matches your original diagram:  
- We still do classification;  
- We still do a prompt to see if the user wants to re-run cached code or ignore it;  
- We still do `docker-compose` flows for code execution, etc.
---


## Key Points / Summary

- **Separated** responsibilities into modules:
  - `session_manager.py` for session context  
  - `faiss_cache.py` for database caching + FAISS  
  - `docker_executor.py` for Docker Compose updates/cleanup  
  - `prompts.py` for all LLM interactions: classification, refinement, final pass  
  - `run_code.py` for actually building & running user code in Docker  
  - `app.py` for the main loop orchestrating everything.  

- **Maintained** the same docstrings and code logic where possible.  
- **Refined** the code to remove repeated logic and unify certain flows.  
- **Still** uses an interactive approach matching your original flowchart.  

You can adapt file names or combine certain modules if you prefer fewer files, but this setup provides a clear example of how to implement all your improvements while honoring your original design.