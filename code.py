import requests

url = "https://connection.keboola.com/v2/storage/components/keboola.python-transformation-v2/configs"

header = {
	"X-StorageApi-Token": "9832-638325-OeNl9pQN0TKGlYaAetk7BWu4tNnZIORKHxNtCknG",
	"Content-Type":"application/json"
}

from prettytable import PrettyTable
r = requests.get(url, headers=header)
#data_in =r.json()
#print(data_in)

dictionary = r.json()

headers = []
values = []
for key in dictionary:
    head = key
    value = ""
    print (key)
 
'''
 if type(dictionary[key]) == type({}):
        for key2 in dictionary[key]:
            head += "/" + key2
            value = dictionary[key][key2]
            headers.append(head)
            values.append(value)

    else:
        value = dictionary[key]
        headers.append(head)
        values.append(value)
'''
print(headers)
print(values)
myTable = PrettyTable(headers)

myTable.add_row(values)
print(myTable)
