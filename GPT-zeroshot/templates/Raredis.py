#%% libraries
from universal_classes import Entity, Relation
from utils import recursive_lowercase
path_system_prompt = 'prompts/Raredis.txt'
with open(path_system_prompt, 'r') as file:
    system_prompt = file.read()

class RaredisTemplate_json:

    @classmethod
    def extract_relations(cls, relation_list):
        relations = set()
        relation_list = recursive_lowercase(relation_list)
        for el in relation_list:
            entity1 = Entity({el['entity1']['text']}, el['entity1']['entity_type'])
            entity2 = Entity({el['entity2']['text']}, el['entity2']['entity_type'])
            relation = Relation({entity1, entity2}, el['relation'])
            relations.add(relation)

        return relations

    @classmethod
    def make_prompt(cls, example):
        system_content = system_prompt

        user_content = f'Now extract all rare disease relations from the following biomedical text. \n\n Text: {example.text} '

        system = {'role': 'system',
                  'content': "You are a helpful assistant designed to output JSON. " + system_content}

        user = {'role': 'user',
                'content': user_content}

        messages = [system, user]

        return messages