import os

from tqdm import tqdm

from lxml import etree

TABLE_SIZE = 10


# ---------------------------------------------------------------------------------------------------------------
def get_hierarchical_attrib(xml_obj):
    attribs = list()
    for node in xml_obj.iter():
        attrib = node.attrib
        if len(attrib) != 0:
            for k in attrib.keys():
                tag = node.tag[node.tag.find('}') + 1:]
                prop = k[k.find('}') + 1:]
                if attribs.count(tag) == 0:
                    attribs.append(tag)
    return attribs


# ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    property_all = [set(), set()]  # first is benign, second is mal
    benign_path = r'C:\-\-\-\-\-'
    mal_path = r'C:\-\-\-\-\-'
    root_path = [benign_path, mal_path]

    number_of_file = 4294949494

    foo = 0
    for root in root_path:
        temp = 0
        for path, dirs, files in os.walk(root):
            for file in files:
                if file.count("xml") != 0:
                    if temp == number_of_file:
                        break
                    temp += 1
                    file_path = os.path.join(path, file)
                    parsed_xml = etree.parse(file_path, parser=etree.XMLParser(huge_tree=True))
                    parsed_xml_list = get_hierarchical_attrib(parsed_xml)
                    for each in parsed_xml_list:
                        property_all[foo].add(each)
        foo += 1

    for i in range(2):
        property_all[i] = list(property_all[i])

    prop_intersection = []
    prop_union = [item for item in property_all[1]]
    for each in property_all[0]:
        if property_all[1].count(each) != 0:
            prop_intersection.append(each)
        else:
            prop_union.append(each)

    with open('union.txt', 'w', encoding='utf-8') as f:
        for each in prop_union:
            f.write(each + '\n')

    with open('intersection.txt', 'w', encoding='utf-8') as f:
        for each in prop_intersection:
            f.write(each + '\n')

    with open('only_mal.txt', 'w', encoding='utf-8') as f:
        for each in property_all[1]:
            if prop_intersection.count(each) == 0:
                f.write(each + '\n')

    with open('only_benign.txt', 'w', encoding='utf-8') as f:
        for each in property_all[0]:
            if prop_intersection.count(each) == 0:
                f.write(each + '\n')
