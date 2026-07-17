
import sys, requests, re, textwrap, ast, time, csv

from concurrent.futures import ThreadPoolExecutor
from requests.auth  import HTTPBasicAuth

from rdflib import Graph, Namespace

from src_en.utils      import diff_time
from src_en.config     import settings
from src_en.utils.text import normalize

# resposta em turtle
def graph_search(class_rules,expantion,keyword,question,named_graph,retrieval_size):

    print( f'\n--> search graph ({settings.repositorio})')

    status         = ''
    filter         = ''
    knowledge_base = {}
    resp_final     = resp_rules_toon = resp_chunks_toon  = ""
    filter_article = filter_chapter  = ""    
    keyword        = re.sub(r'[\\/]+', ' ', keyword)  
    keyword        = re.sub(r'([\\\'"])', r'\\\1', keyword)
    article        = expantion['query_expansion']['article']
    chapter        = expantion['query_expansion']['chapter'] 
    class_rules_lucene = ", ".join(class_rules.split())  

    if article and article.strip().lower() not in ('none', 'null'):
        article = article.lower()
        filter_article = f'CONTAINS(LCASE(STR(?bc)), "{article}")'

        articles = [
            a.strip().lower()
            for a in article.split('|')
            if a.strip()
        ]

        conditions = [
            f'CONTAINS(LCASE(STR(?bc)), "{a}")'
            for a in articles
        ]

        filter_article = " || ".join(conditions)        
        retrieval_size = 50

    if chapter and chapter.strip().lower() not in ('none', 'null'):
        chapter = chapter.lower()
        filter_chapter = f'CONTAINS(LCASE(STR(?bc)), "{chapter}")' 

        chapter = [
            a.strip().lower()
            for a in chapter.split('|')
            if a.strip()
        ]

        conditions = [
            f'CONTAINS(LCASE(STR(?bc)), "{a}")'
            for a in chapter
        ]

        filter_chapter = " || ".join(conditions)        
        retrieval_size = 50   

    if filter_article:
        filter = f'FILTER( {filter_article} )'

    if filter_chapter:
        filter = f'FILTER( {filter_chapter} )'   

    if filter_article and filter_chapter:
        filter = f'FILTER( {filter_article} && {filter_chapter} )' 

    try:

        query_class_rules = f'''
            PREFIX :     <https://omc.co/vocabulary/>
            PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            CONSTRUCT {{
                ?regraURI rdf:type ?tipo .
                ?regraURI :descricao ?descricao .
                ?regraURI :texto ?texto .
                ?regraURI :breadcrumb ?breadcrumb .
                ?regraURI :sourceChunk ?chunk .
            }}

            FROM <https://omc.co/graph/5511993891773>
            WHERE {{
                VALUES ?classe {{
                    {class_rules}
                }}

                # 1. Instância da classe extraída pelo LLM
                ?instancia rdf:type ?classe .

                # 2. Chunk hub conectado à instância (em qualquer direção)
                {{
                    ?instancia :relacionamento ?chunk .
                }} UNION {{
                    ?chunk :relacionamento ?instancia .
                }}
                ?chunk rdf:type :Chunk .

                # 3. Regras conectadas ao mesmo chunk hub
                {{
                    ?regraURI :relacionamento ?chunk .
                }} UNION {{
                    ?chunk :relacionamento ?regraURI .
                }}
                ?regraURI rdf:type/rdfs:subClassOf* :Rule .

                OPTIONAL {{ ?regraURI :descricao ?descricao . }}
                OPTIONAL {{ ?regraURI rdf:type ?tipo }}
                OPTIONAL {{ ?chunk :texto ?textoBruto . }}
                OPTIONAL {{ ?chunk :idChunk ?breadcrumb }}

                BIND(COALESCE(?textoBruto, ?descricao) AS ?texto)

            }}
            ORDER BY ?regraURI
            Limit 50
        '''
         
        query_lucene = f'''
            PREFIX : <https://omc.co/vocabulary/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX luc: <http://www.ontotext.com/connectors/lucene#>
            PREFIX luc-index: <http://www.ontotext.com/connectors/lucene/instance#>

            CONSTRUCT {{ ?uri rdf:type ?tipo . ?uri :fullText ?texto . ?uri :breadcrumb ?bc . ?uri :score ?score . }}
            FROM <https://omc.co/graph/5511993891773>
            WHERE {{
                {{
                    ?uri rdf:type :Rule .        
                    FILTER (?obj IN ({class_rules_lucene}))
                    OPTIONAL {{ ?uri :descricao ?desc . }}
                    OPTIONAL {{ ?uri :title ?tit . }}
                    BIND(COALESCE(?desc, ?tit, "") AS ?texto)
                    BIND(:Rule AS ?tipo)
                    BIND(0 AS ?score)

                }} UNION {{
                    ?q a luc-index:eu_full ;
                    luc:query "{keyword}" ;
                    luc:entities ?uri .
                    ?uri luc:score ?score .
                    ?uri rdf:type :Point .
                    OPTIONAL {{ ?uri :fullText ?texto . }}
                    OPTIONAL {{ ?uri :breadcrumb ?bc . }}
                    {filter}
                    BIND(:Point AS ?tipo)

                }}
            }}
            ORDER BY DESC(?score) ?uri
            LIMIT 50
        '''

        url     = f"{settings.GRAPHDB_BASE_URL}/repositories/{settings.repositorio}"
        headers = {"Content-Type": "application/sparql-query", "Accept": "text/turtle"}

        # Buscar apenas Regras
        resp_rules = requests.post(
            url,
            data=query_class_rules,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME,settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )

        if resp_rules.status_code == 200:
            
            V = Namespace("https://omc.co/vocabulary/")

            g = Graph()
            g.parse(data=resp_rules.text, format="turtle")

            '''with open('_retrieval.txt', mode='w', encoding='utf-8') as f:
                f.write('Rules\n')
                f.write(resp_rules.text)'''

            for uri in set(g.subjects()):
                
                breadcrumb_node = next(g.objects(uri, V.breadcrumb), None)
                score_node = next(g.objects(uri, V.score), None)
                texto_node = (
                    next(g.objects(uri, V.fullText), None)
                    or next(g.objects(uri, V.texto), None)
                    or next(g.objects(uri, V.descricao), None)
                )

                breadcrumb = str(breadcrumb_node or uri)
                score = round(float(score_node), 3) if score_node else 0
                texto = normalize(str(texto_node or ""))

                knowledge_base[breadcrumb] = texto

        else:

            error_msg = f"GraphDB Rule ERROR ({resp_rules.status_code}): {resp_rules.text}"
            print(f"--> {error_msg}")
            print('---- rules ----') 
            print( query_class_rules )

            sys.exit()   
    
            return {
                'status': 'ERROR', 
                'response': error_msg, 
                'dataset': {}
            }         
                
        #schema = extract_instances()
        #query  = auto_query('ollama',schema,question,class_rules,keyword,filter)
        #print( query_lucene )
        
        headers = {"Content-Type": "application/sparql-query", "Accept": "text/turtle"}

        resp_chunks = requests.post(
            url,
            data=query_lucene,
            headers=headers,
            auth=HTTPBasicAuth(settings.GRAPHDB_USERNAME, settings.GRAPHDB_PASSWORD)  # ajuste usuário/senha
        )
   
        if resp_chunks.status_code == 200:
            
            V = Namespace("https://omc.co/vocabulary/")

            g = Graph()
            g.parse(data=resp_chunks.text, format="turtle")

            '''with open('_retrieval.txt', mode='a', encoding='utf-8') as f:
                f.write('Chunk\n')
                f.write(resp_chunks.text)'''

            for uri in set(g.subjects()):
                
                breadcrumb_node = next(g.objects(uri, V.breadcrumb), None)
                score_node = next(g.objects(uri, V.score), None)
                texto_node = (
                    next(g.objects(uri, V.fullText), None)
                    or next(g.objects(uri, V.texto), None)
                    or next(g.objects(uri, V.descricao), None)
                )

                breadcrumb = str(breadcrumb_node or uri)
                score = round(float(score_node), 3) if score_node else 0
                texto = normalize(str(texto_node or ""))

                knowledge_base[breadcrumb] = texto            
                
        else:
            
            error_msg = f"GraphDB lucene ERROR ({resp_chunks.status_code}): {resp_chunks.text}"
            print(f"--> {error_msg}")
            print('---- chunks ----') 
            print( query_lucene )

            sys.exit()   
    
            return {
                'status': 'ERROR', 
                'response': error_msg, 
                'dataset': {}
            } 
        
        resp_final = f'''Deontonic rules:
        {textwrap.dedent(resp_rules.text)}
        General Context:
        {textwrap.dedent(resp_chunks.text) }
        '''          

        return {
            'status':'OK',
            'response':resp_final,
            'dataset':knowledge_base
        }
    
    except Exception as e:

        error_msg = f"Graph ERROR: {str(e)}"
        print(f"--> {error_msg}")
        print(f"--> {error_msg}")

        sys.exit()
        
        return {
            'status': 'ERROR', 
            'response': error_msg, 
            'dataset': {}
        }
