# Stole some of this from a StackOverflow, but forgot where



from xml.etree import cElementTree as ElementTree
from xml.etree import ElementTree as ET

# Stolen
class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)

# Stolen
class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def read_xmlphases( xmlfile ):

    import pandas as pd
    from obspy import UTCDateTime

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    phasepicks = []

    for a in root[0]:
        ##print(a)
        ##print(a.tag)
        ##print(a.attrib)

        if 'pick' in a.tag:
            #print('>>> Pick')
            #print(a[0])
            for b in a:
                ##print(b)
                ##print(b.tag)
                ##print(b.attrib)
                ##print(b.text)
                if 'time' in b.tag:
                    print('   >>> Time')
                    for c in b:
                        ##print(c)
                        ##print(c.tag)
                        ##print(c.attrib)
                        print(c.text) # <- This is the time of the pick
                        time = UTCDateTime(c.text)
                        ##print()
                elif 'waveformID' in b.tag:
                    print('   >>> waveformID')
                    ##print(b)
                    ##print(b.tag)
                    print(b.attrib) # <- dict e.g., {'networkCode': 'VV', 'stationCode': 'COP', 'locationCode': 'CP', 'channelCode': 'HHZ'}
                    #nslc = b.attrib
                    nslc = '{}.{}.{}.{}'.format(b.attrib['networkCode'], b.attrib['stationCode'], b.attrib['locationCode'], b.attrib['channelCode'])
                    ##print(b.text)
                elif 'phaseHint' in b.tag:
                    print('   >>> phaseHint')
                    print(b.text) # <- Phase Hint
                    phase = b.text
            phasepicks.append(dict({'time':time, 'nslc': nslc, 'phase':phase}))
            print()

    df = pd.DataFrame(phasepicks)
    return df
