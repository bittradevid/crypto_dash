import requests

import ssl

# Crear un contexto SSL que no verifique certificados
ssl._create_default_https_context = ssl._create_unverified_context

import requests

url = "https://liquidation-report.p.rapidapi.com/lickhunterproplus"

headers = {
	"x-rapidapi-key": "6078415a4fmshda1f34dd3581f68p1281d5jsn66cfbd8ce9c5",
	"x-rapidapi-host": "liquidation-report.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())