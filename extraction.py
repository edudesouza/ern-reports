
import langextract as lx

from sentence_transformers import util

from src_en.evaluation import keyword_complexity
from src_en.config import settings

from src_en.models import embedding_model

def keywords_create(question,model,api,url):

    ontology_txt=''
   
    with open(settings.ontology, encoding="utf-8") as f:
        ontology_txt = f.read()
    
    prompt = f'''
        ## 1. FUNDAMENTAL CONCEPTS
        You are an expert in Ontologies and Intent Extraction.
        Your task is to analyze the user's question and extract structured data based on the provided Ontology.
        
        ## 2. ONTOLOGY REFERENCE
        {ontology_txt}

        ## 3. EXTRACTION INSTRUCTIONS (5W3H METHODOLOGY)
        Analyze carefully to identify the main item being asked about.
        For the main topics, always look for at least 2 synonyms.
        We will use 5W3H, which is a structured questioning methodology designed to organize thinking and action planning.
        Acronym    Question               Practical function
        What       What will be done?     Defines the objective or task.
        Why        Why will it be done?   Defines the purpose or justification.
        Where      Where will it be done? Defines the place or context.
        When       When will it be done?  Defines the deadline or schedule.
        Who        Who will do it?        Defines the responsible party.
        How        How will it be done?   Defines the method or process.
        How much   How much will it cost? Defines the cost or required resources.
        How many   How many resources?    Defines the quantity or scale.
        If necessary, create more than one set.
        If you do not find all 5W3H items, return only NULL in the item where you did not find the facts.
        Do not include stop words.
        Do not use double quotes or single quotes.
        Do not use slashes, pipes, or backslashes: '/','\','|'; instead, write them textually using: 'or','and'.
        
        ## CANONICAL ##
        Also generate a canonical query with a maximum of 12 terms, using only nouns and main verbs, without generic terms.
        
        ## ARTICLE AND CHAPTER ##
        Extract article and chapter ONLY if they are explicitly mentioned in the user's question otherwise mark as none.
        Do NOT infer article or chapter from the law name, topic, legal concept, or prior knowledge, if not present mark as none.
        Identify if the question has writen a specific article, if note mark as none
        Identify if the question has writen a specific chapter, if note mark as none
        In cases that you find more than one article or chapter, separate them with pipe, for example: 'article 2|article 5' or 'chapter i|chapter ii'.
        Avoid duplicates.

        ## CLASSES ##
        Analyze the user's question
        Aanalyze the rewritten and expanded question with as broader context
        Analyze the canonical fact that represents the main legal point to be answered
        Return the related classes thaart e present in the ontology        

        Returns only the classes without any comments, or any extra info
        Always return plain text

        Example:
        :SensitivePersonalData :GivenConsent :Controller

        - DO NOT create any class that is not presente in the ontology
        - The soure should be only and always the given ontology               
        - As a result we will deliver only classes present in the ontology
    '''

    examples = [
        lx.data.ExampleData(
            text="My 15-year-old daughter's school requires facial recognition for entry and attendance. Students who refuse cannot enter. The system is operated by an external company, and I do not know where the data is stored or how it is used.",
            extractions=[
                lx.data.Extraction(
                    extraction_class = "triple",
                    extraction_text  = "I want to know until what time I can use the pool.",
                    attributes       = 
                    {
                        "what":     "Mandatory facial recognition for entry and attendance.",
                        "why":      "To control access and attendance.",
                        "where":    "At the school entrance. Data storage is unknown.",
                        "when":     "At entry and during attendance checks.",
                        "who":      "A 15-year-old student, the school, and an external company.",
                        "how":      "By collecting and processing facial biometric data.",
                        "how_much": "The amount of data collected is unknown.",
                        "how_long": "The data-retention period is unknown.",
                        "canonical":"Mandatory facial recognition of a minor for school access and attendance, operated by an external company with unknown data practices.",
                        "article":  "article 2",
                        "chapter":  "chapter (i,1)",
                        "class":":  SensitivePersonalData :GivenConsent :Controller"
                    },
                )
            ],
        )
    ]

    try:

        res_ex = lx.extract(
            text_or_documents=question,
            prompt_description=prompt,
            examples=examples,
            model_id=model, 
            api_key=api,
            model_url=url,  
            fence_output=False,
            use_schema_constraints=True,
        )
    
    except Exception as erro:
        
        print( f'ERRO: {erro}' )
        exit()
        
        return

    triples = []

    for ext in getattr(res_ex, "extractions", []):
        if ext.extraction_class == "triple":
            triples.append(ext.attributes)

    palavras_chave = ''

    #complexity = keyword_complexity(triples)
    #print( f'--> complexidade {complexity}')

    components = []
    expansion  = {}

    #valid_parts = [v for t in triples for v in t.values() if str(v).upper() != 'NULL']
    #keywords    = ", ".join(valid_parts)

    allowed_parts = ["what", "why", "where", "when", "who", "how", "how_much", "how_many"]

    valid_parts = [
        t.get(part)
        for t in triples
        for part in allowed_parts
        if str(t.get(part)).upper() != "NULL"
    ]

    keywords = ", ".join(valid_parts)

    for t in triples:  

        components = [
            t.get('what'),
            #t.get('why'),
            #t.get('where'),
            #t.get('when'),
            #t.get('who'),
            t.get('how'),
            #t.get('how_much'),
            #t.get('how_many')
        ]

        expansion = {
            "what":t.get('what'),
            "why":t.get('why'),
            "where":t.get('where'),
            "when":t.get('when'),
            "who":t.get('who'),
            "how":t.get('how'),
            "how_much":t.get('how_much'),
            "how_many":t.get('how_many'),
            "canonical":t.get('canonical'),
            "article":t.get('article'),
            "chapter":t.get('chapter'),
            "class":t.get('class')

        }

        '''print( f' what:     {t.get('what')}' )
        print( f' why:      {t.get('why')}' )
        print( f' where:    {t.get('where')}' )
        print( f' when:     {t.get('when')}' )
        print( f' who:      {t.get('who')}' )
        print( f' how:      {t.get('how')}' )
        print( f' how_much: {t.get('how_much')}' )
        print( f' how_many: {t.get('how_many')}' )'''

    # Calcular embeddings
    embeddings = embedding_model.encode(components)

    # Matriz de similaridade
    similarities = util.cos_sim(embeddings, embeddings)

    # Análise
    avg_similarity = similarities.mean().item()
 
    return {
        "keywords":keywords,
        "complexity_score":avg_similarity,
        "query_expansion":expansion
    }
