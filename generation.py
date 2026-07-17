
import json,ast

from langchain_openai                import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama                import ChatOllama
from langchain_anthropic             import ChatAnthropic
from langchain_google_genai          import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk

from src_en.config import settings

def response_create(keyword,question,context,rules,canonical,model_provider): 

    if model_provider=='maritaca':
        llm = ChatMaritalk(
            model='sabia-4', 
            api_key=settings.MARITACA_API_KEY,
            temperature=0.1, 
            max_tokens=1000, 
            model_kwargs={"response_format": {"type": "json_object"}}
        ) 

    elif model_provider=='google':
        llm = ChatGoogleGenerativeAI(
            #model="gemini-3.1-flash-lite",
            model="gemini-2.5-flash-lite",
            #model="gemma-4-31b-it",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        ).bind(
            response_mime_type="application/json"
        )  

    elif model_provider=='gpt':
        llm = ChatOpenAI(
            model='gpt-4.1', 
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        ) 

    elif model_provider=='claude':
        llm = ChatAnthropic(
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
            model="claude-sonnet-4-6"
        )

    elif model_provider=='ollama':
        llm = ChatOllama(  

            #model="kimi-k2-thinking:cloud",
            #model="kimi-k2.6:cloud",

            #model="deepseek-v3.2:cloud",
            #model="deepseek-v3.1:671b-cloud",
            #model="deepseek-v4-pro:cloud", 
                       
            #model="qwen3-next:80b-cloud",
            #model="qwen3.5:397b-cloud",
            #model="qwen3.5:0.8b",                   #pequeno

            #model="gemini-3-pro-preview:latest",
            model="gemma4:cloud",
            #model="gemma3:27b-cloud",            
            #model="gemma3:4b-cloud",            

            #model="mistral-large-3:675b-cloud",
            #model="ministral-3:3b-cloud",           #pequeno

            #model="gpt-oss:20b-cloud",
            #model="lfm2.5-thinking:1.2b",
            
            #model="minimax-m2.1:cloud",                  
            #model="minimax-m2:cloud",

            num_predict=1024,
            temperature=0,
            format="json",
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

    else:
        raise ValueError(f"Unsupported model_provider for response_create: {model_provider!r}")

    print(f'--> create llm response, {model_provider} {getattr(llm, "model", None)}, temp.: {getattr(llm, "temperature", None)}')

    with open(settings.ontology, encoding="utf-8") as f:
        ontology_txt = f.read()

    system = f'''You are a legal assistant for law interpretation and explanations  
        Do not invent juridical relations outside the ontology.
        Prefer rule-centric reasoning over semantic approximation.

        Answer ONLY based on the evidence provided.
        If there is an explicit rule, cite it.
        If there is not sufficient basis, state so.

        Relationships: show attributes that demonstrate the relationship between the items used in the answer.

        **ABSOLUTE RULES:**
        1. Never start the answer with: yes, no, of course, for sure, negative, positive
        2. The "answer" field MUST have maximum of 900 characters, prioritizing minimal and precise answers, while preserving completeness. Avoid verbosity and do not add information beyond what is necessary to answer the question.
        3. The "full_answer" field has a limit of 2000 characters
        4. OUTPUT: Pure, valid JSON, without markdown (no ```json or ```)

        **IMPORTANT**:   
        - Answer only using the provided graph evidence.
        - If the answer is not explicitly supported by the graph nodes, say: "I could not find sufficient evidence in the graph."
        - Do not use prior knowledge.
        - Preserve modal verbs exactly: shall, may, must.
        - Cite the node IDs used.
        - Response should NOT have break line.
        - Always return a valid JSON.
                 
        **GROUNDING**:  
        Do NOT infer a law name, topic, legal concept, or prior knowledge, if not present in current context
        The "breadcrumb" is the legal path that leads to the answer, it is the "trail" of legal concepts and rules that support the answer. It is mandatory to include the breadcrumb in the response, as it will be used to evaluate the accuracy and precision of the answer.
        As avidence of a response you can only use the "breadcrumb" 
    '''
    
    user = f'''
        You are a legal assistant specializing in law interpretation and explanations

        Question or doubt:
        {question}

        Facts that must be answered:
        {keyword}

        Broad context:
        {context}

        Related normative rules       
        {rules}

        # LEGAL ANALYSIS METHODOLOGY

        ## 1. FUNDAMENTAL CONCEPTS
        - **Key passage**: Specific textual excerpt from the document that legally supports your answer
        - **Multi-chunk analysis**: Ability to synthesize information from multiple documents
        - **Hierarchy of relevance**: Score indicates source reliability
        - Never use **depends** in your answers
        - The answer MUST have **maximum of 900 characters**, you must rewrite if necessary

        ## 2. ANALYTICAL PROCESS (execute in this order)

        ### Phase 1: Understanding
        - Identify the legal nature of the question (normative, interpretative, consultative)
        - Map key concepts and legal terms Relevant
        - Consider practical implications and possible exceptions

        ### Phase 2: Investigation in the Chunks
        - Prioritize the rules and then the chunks, type: 'Rule' or 'Chunk'
        - Prioritize chunks with a _score > 0.85 as primary sources
        - From the user's question, create a premise, carefully analyze the context,
        - Seek a logical conclusion for this premise; this conclusion can be affirmative or negative
        - You may find facts and arguments in the context that justify or refute what is being asked
        - Identify complementary rules or chunks (0.60-0.85) for additional context
        - Be aware of ambiguities, as in one rule or chunk you may have wording that conflicts with another
        - How to proceed with disambiguation:
        -- Analyze the question, determining the real motivator of what is being questioned
        -- Analyze which of the candidate options best meets the requirements that answer the question
        -- Look for logical relationships between different chunks and rules:

        * General rule + exceptions
        * Right + corresponding obligation
        * Norm + penalty
        * Permission + requirements

        ### Phase 3: Legal Synthesis

        - Construct a response that integrates multiple chunks when applicable
        - Identify the most relevant excerpt (up to 500 characters)
        - Provide a clear conclusion: YES or NO, and explicitly state the conditions or evidence that support that conclusion.
        - Maintain accessible language without losing technical precision

        ## 3. REASONING PRINCIPLES
        - **Controlled Literalness**: Base yourself strictly on the text, but interpret it within a legal context
        - **Information Economy**: Prioritize quality over quantity of chunks
        - **Transparency**: Explicitly state when there is ambiguity or insufficient information
        - **Intelligent Connections**: Relate chunks that address the same topic from different angles

        ## 4. MANDATORY VALIDATIONS
        - [ ] All chunk IDs exist in the provided context
        - [ ] Percentages add up to 100% and reflect the real contribution of each chunk
        - [ ] Key passage is a verbatim quote (not a paraphrase)
        - [ ] Answer has a logical consequence based on the chunks
        - [ ] No information was invented or inferred without a textual basis
        - [ ] The answer MUST have maximum of 900 characters

        ## 5. Control Mechanism:
        - If any of the validations fail, rewrite the answer correcting the error.
        - The answer MUST have maximum of 1000 characters; if the answer has fewer or more characters, discard it and generate a new one.        
    '''

    draft       = llm.invoke([("system", system), ("user", user)])
    draft_clean = draft.content.replace('```json', '').replace('```', '').strip()

    print( '--- darft ---')
    print( draft_clean )
    print( '-'*100 )

    harness_system = '''You are a normative compliance verifier.
        Your role is to audit a GENERATED ANSWER against the supplied RULES AND EVIDENCE. Do not answer from prior knowledge.

        ## EVIDENCE RULES
        1. Use only the textual content of the supplied RULES AND EVIDENCE.
        2. Treat breadcrumbs and node IDs only as source identifiers.
        3. Every legal claim must be supported by the evidence.
        4. Do not infer any law, concept, relationship, obligation, permission, prohibition, exception, responsibility, or consequence not established by the evidence.
        5. Preserve the meaning of normative modal verbs, including “shall”, “must”, and “may”.
        6. Preserve legally relevant actors, actions, objects, conditions, exceptions, restrictions, scopes, qualifiers, thresholds, temporal requirements, references, and delegations.
        7. Do not replace a qualified legal condition with a broader semantic category.
        8. Do not treat mandatory facts or canonical facts as legal evidence. Use them only to evaluate whether the answer covers the question.

        ## AUDIT CRITERIA
        Determine whether the generated answer:
        1. Directly answers the question.
        2. Is fully supported by the evidence.
        3. Covers the mandatory facts and canonical fact.
        4. Preserves all relevant conditions, exceptions and qualifiers.
        5. Contains unsupported or externally derived claims.
        6. Converts an exception into a general rule.
        7. Converts a necessary condition into a sufficient condition.
        8. Converts a delegation into an operational rule.
        9. Infers an unsupported obligation, permission, prohibition, responsibility or consequence.
        10. Compresses a qualified provision into an overbroad category.
        11. Omits a necessary part of an exhaustive enumeration.
        12. Contradicts the evidence.

        ## COMPLETENESS RULE
        A summary is acceptable only if it preserves the legal meaning and scope of the evidence.
        An omission is material if it broadens or narrows the rule, removes a condition or exception, changes an actor or modality, removes a legally relevant relationship, or presents a conditioned category as an independent rule.
        Do not treat the omission of nonessential examples as material when the governing category is accurately and completely preserved.

        ## DECISION RULES
        Return “APPROVE” when the answer is supported, correct, sufficiently complete and normatively precise.
        Return “REVISE” when the central conclusion is correct but the answer contains a relevant omission, imprecision, unsupported generalization or loss of a legal qualifier.
        Return “REJECT” when the central conclusion is unsupported, contradicted by the evidence or based on invented legal information.
        
        For “APPROVE”, corrected_answer must be null.
        For “REVISE”, provide a corrected answer based only on the evidence.
        For “REJECT”, provide a corrected answer only if the evidence is sufficient. Otherwise, corrected_answer must be null.

        ## OUTPUT
        Return only valid JSON without Markdown or line breaks       
        **Required format**:
        {{
            "answer": "Direct and grounded response (maximum of 900 characters). Cite the rule or norm. Do not include chunk IDs here.",
            "full_answer": "Detailed legal analysis (up to 2000 characters). Expand reasoning, exceptions, and implications.",
            "justification": "Concise explanation of why the original generated response was approved, revised or rejected. Identify the decisive findings from the audit without introducing new legal claims.",
            "lost_conditions": ["List each legally relevant condition, exception, qualifier, scope limitation, relationship or delegation that was omitted, altered or overgeneralized in the original generated response. Return an empty list when none is found."],            
            "key_snippet": "Verbatim excerpt from the document that legally supports the answer (up to 2000 characters).",
            "chunks": ["breadcrumb_1", "breadcrumb_1"],            
            "decision":"revise|approve|reject",
            "percentage": ["80", "20"]
        }}
    '''    

    harness_user = f'''You are a normative compliance verifier.
        QUESTION:
        {question}

        FACTS THAT MUST BE ANSWERED:
        {keyword}

        CANONICAL FACT TO BE COVERED:
        {canonical}

        GENERATED ANSWER:
        {draft_clean}

        RULES AND EVIDENCE:
        {rules}

        Audit the generated answer and return the required JSON.

        ## OUTPUT FORMAT
        - Your interpretation MUST have maximum of 1000 characters, never more than 2000 characters.
        - Return only JSON without markdown, following the exact structure specified in the example below:

        {{
            "answer": "preserve de original answer",
            "full_answer": "preserve de original full answer",
            "justification": "explain the decision",
            "lost_conditions": "show what is missing in the answer",
            "key_snippet": "import textual partition",
            "chunks": ["<breadcrumb_1>", "<breadcrumb_2>"],
            "decision":"<revise|approve|reject>",
            "percentage": ["<80>", "<20>"]
        }}        

        **IMPORTANT**:        
        GROUNDING: You can only use information to respond, present in this context
        VALIDATION: if the answer has more than 900 characters, rewrite it to meet the *MANDATORY* character limit (maximum of 900), maintaining the essence.
    '''

    msg = llm.invoke([("system", harness_system), ("user", harness_user)])

    #print(f'tokens: {msg.usage_metadata}\n' )
    '''print('-'*100)
    print(user_v1)   
    print('-'*100) '''

    try:

        print('--> llm response OK!')
        print( msg.content )

        texto_limpo = msg.content.replace('```json', '').replace('```', '').strip()
        resp_json   = json.loads(texto_limpo) 

        return resp_json 
    
    except Exception as e:

        print('--> llm response ERRO!')
        print( f"Erro ao processar resposta do modelo: {e} {msg}" )
        print( msg.content )
        print( '-'*100 )

        return {
            "resposta": f"Erro ao processar resposta do modelo: {e} {msg}"
         }
