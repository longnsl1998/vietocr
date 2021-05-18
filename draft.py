import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\ASUS\\DM\\Cosmetics.csv")

def Supp_K_itemset(itemSet, listData):  # support của mỗi candidate k-itemset
    count = 0

    for data in listData:
        add = 1
        for i in itemSet:
            if data[i] == 0:
                add = 0
                break
        count += add

    return count / len(listData)


def join_itemset(itemset1, itemset2):  # Lk-1 nối Lk-1  để sinh ra candidate k-itemset
    new = []
    if len(itemset1) != len(itemset2):
        return None

    if len(itemset1) == 1:
        new.append(itemset1[0])
        new.append(itemset2[0])
        return new

    new.append(itemset1[0])
    for i in range(1, len(itemset1)):
        if itemset1[i] != itemset2[i - 1]:
            return None
        new.append(itemset1[i])
    new.append(itemset2[len(itemset2) - 1])

    return new


def frequent_itemsets(data, minSupp):  # Tìm frequent itemset I
    listItemSets = []
    listSupp = []
    botIndex = 0
    numberNewItemsSet = 0

    for i in range(len(data[0])):
        itemset = [i]
        supp = Supp_K_itemset(itemset, data)
        if supp >= minSupp:
            listItemSets.append(itemset)
            listSupp.append(supp)
            numberNewItemsSet += 1

    old = numberNewItemsSet

    while numberNewItemsSet != 0:
        numberNewItemsSet = 0
        top = len(listItemSets)
        for i in range(botIndex, top):
            for j in range(i + 1, top):
                itemset = join_itemset(listItemSets[i], listItemSets[j])
                # print(itemset)
                if itemset is None:
                    continue
                supp = Supp_K_itemset(itemset, data)
                if supp >= minSupp:
                    listItemSets.append(itemset)
                    listSupp.append(supp)
                    numberNewItemsSet += 1
        botIndex += old
        old = numberNewItemsSet

    return listItemSets, listSupp


class Rule:
    premise = []
    consequence = []
    supp = 0.0
    conf = 0.0

    def __init__(self, premise, consequence, supp):
        self.premise = premise
        self.consequence = consequence
        self.supp = supp

    def SetConf(self, conf):
        self.conf = conf


def GenRule(rule, minConf, listItemSets, listSupp, listRules):  # sinh ra các luật  với conf > =min_conf
    premise = rule.premise.copy()
    consequence = rule.consequence.copy()
    if len(rule.premise) > 1:
        for i in rule.premise:
            premise.remove(i)
            consequence.append(i)
            newRule = Rule(premise, consequence, rule.supp)
            premise = rule.premise.copy()
            consequence = rule.consequence.copy()
            # print(newRule.premise)
            # print(newRule.consequence)
            # print(newRule.supp)
            conf = (newRule.supp) / (listSupp[listItemSets.index(newRule.premise)])
            if conf < minConf:
                continue
            # print(newRule.supp)
            # print(listSupp[listItemSets.index(newRule.premise)])
            # print(conf)
            newRule.SetConf(conf)
            listRules.append(newRule)
            GenRule(newRule, minConf, listItemSets, listSupp, listRules)
    # print(listRules)


def List_Rules(listItemSets, listSupp, minConf):  # List Rules
    listRules = []

    for itemset in listItemSets:
        if len(itemset) == 1:
            continue
        rule = Rule(itemset, [], listSupp[listItemSets.index(itemset)])
        GenRule(rule, minConf, listItemSets, listSupp, listRules)

    return listRules


def Apriori(listname, data, minSupp, minConf):
    listItemsSet, listSupp = frequent_itemsets(data, minSupp)
    listRules = List_Rules(listItemsSet, listSupp, minConf)
    print(listItemsSet)
    print("num_Rules : ", len(listRules))
    print("__________________________________________________________________________________________")
    for rule in listRules:
        # print(rule)
        ruleString = ""
        for i in rule.premise:
            ruleString += listname[i] + ";"
        ruleString += "=>"
        for i in rule.consequence:
            ruleString += listname[i] + ";"

        ruleString += "  Supp = " + str(rule.supp) + "  Conf = " + str(rule.conf)

        print(ruleString)
        print(str(rule.premise) + " =>   " + str(rule.consequence))
