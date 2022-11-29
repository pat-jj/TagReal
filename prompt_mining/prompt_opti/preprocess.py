import json


def preprocess(relation):
    head_tail_file = '../relation_triples/triples_nyt10' + relation + '.txt'
    prompt_file = '../truepie/output_xy/' + relation + '.txt'

    with open(head_tail_file) as f:
        lines = f.readlines()

    ht_out = []
    for line in lines:
        tmp = {}
        if line != '\n':
            head, tail = line.split('\t')[0], line.split('\t')[2][:-1]
            tmp['pred'] = relation
            tmp['sub'] = head
            tmp['obj'] = tail
            ht_out.append(tmp)
        else:
            break
    

    OUT_FILENAME = f"./input/head_tail/{relation}.jsonl"
    with open(OUT_FILENAME, 'w') as outfile:
        for entry in ht_out:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    with open(prompt_file) as f:
        lines = f.readlines()

    rp_out = []
    for line in lines:
        if line != '\n':
            tmp = {}
            tmp['relation'] = relation
            tmp['template'] = line[:-1]
            rp_out.append(tmp)
        else:
            break

    OUT_FILENAME = f"./input/rel_prompts/{relation}.jsonl"
    with open(OUT_FILENAME, 'w') as outfile:
        for entry in rp_out:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    return 


def main():
    relation_entities = {
        'business_company_founders': {'head': 'COMPANY', 'tail': 'FOUNDER', 'slash': 'business/company/founders'},
        'business_company_place_founded': {'head': 'COMPANY', 'tail': 'PLACE_FOUNDED', 'slash': 'business/company/place_founded'},
        'business_person_company': {'head': 'PERSON', 'tail': 'COMPANY', 'slash': 'business/person/company'},
        'location_administrative_division_country': {'head': 'ADMINISTRATIVE_DIVISION', 'tail': 'COUNTRY', 'slash': 'location/administrative_division/country'},
        'location_country_administrative_divisions': {'head': 'COUNTRY', 'tail': 'ADMINISTRATIVE_DIVISION', 'slash': 'location/country/administrative_divisions'},
        'location_location_contains': {'head': 'LOCATION', 'tail': 'LOCATION_SUB', 'slash': 'location/location/contains'},
        'location_neighborhood_neighborhood_of': {'head': 'LOCATION', 'tail': 'NEIGHBOR', 'slash': 'location/neighborhood/neighborhood_of'},
        'location_us_county_county_seat': {'head': 'US_COUNTY', 'tail': 'COUNTY_SEAT', 'slash': 'location/us_county/county_seat'},
        'people_deceased_person_place_of_death': {'head': 'DECEASED_PERSON', 'tail': 'PLACE_OF_DEATH', 'slash': 'people/deceased_person/place_of_death'},
        'people_ethnicity_people': {'head': 'ETHNICITY', 'tail': 'PEOPLE', 'slash': 'people/ethnicity/people'},
        'people_person_children': {'head': 'PERSON', 'tail': 'CHILDREN', 'slash': 'people/person/children'},
        'people_person_ethnicity': {'head': 'PERSON', 'tail': 'ETHNICITY', 'slash': 'people/person/ethnicity'},
        'people_person_nationality': {'head': 'PERSON', 'tail': 'NATIONALITY', 'slash': 'people/person/nationality'},
        'people_person_place_lived': {'head': 'PERSON', 'tail': 'PLACE_LIVED', 'slash': 'people/person/place_lived'},
        'people_person_religion': {'head': 'PERSON', 'tail': 'RELIGION', 'slash': 'people/person/religion'},
    }

    for relation in relation_entities.keys():
        preprocess(relation=relation)

    return


if __name__ == '__main__':
    main()