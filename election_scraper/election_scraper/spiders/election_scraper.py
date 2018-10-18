import scrapy
import ast
import re
from selenium import webdriver
from scrapy.exporters import BaseItemExporter

class ElectionSpider(scrapy.Spider):
	start_urls = ['http://www.irelandelection.com/election.php?elecid=3&constitid=19&electype=1']
	driver = webdriver.PhantomJS()
	driver.get(start_urls[0])
	name = 'electionspider'

	

	def parse(self, response):

		seats = int(response.css('.well table tr td::text').extract()[0][-1])
		quota = re.search('\d+', str(response.css('.well table tr td::text'). \
						extract()[1]))
		quota = int(quota.group(0))
		election = response.xpath('//select[@name="elecid"]//option[@selected]/text()') \
			.extract_first()
		constit = response.xpath('//select[@name="constitid"]//option[@selected]/text()') \
			.extract_first()

		race_deets = response.xpath('//*[@class = "well"][2]/table/tr/td') \
			.extract()

		race_dict = {'electorate': None, 'turnout': None, 'valid': None, 'spoilt': None}
		year = election[0:4]

		f = open("election_log.txt","w+")
		f.write(constit + " " + year + "\n")
		f.close()


		for r in race_deets:
			cat = str(re.search('\>([a-zA-Z]+)\<', r).group(1)).lower()
			val = int(re.search('\d+', r).group(0))
			race_dict[cat] = val

		cc = len(response.xpath('//*[@id="votesCCtd"]/text()'))

		i = 0
		jscode = str(response.xpath('.//script[@language="javascript"]/text()'). \
			extract_first())
		arr = jscode.split(';')
		transfers = ast.literal_eval(arr[1][arr[1]. \
			find('['):arr[1].find(']]') + 2])
		round_totals = ast.literal_eval(arr[2][arr[2]. \
			find('['):arr[2].find(']]') + 2])

		if cc>0:
			cc = ['CC']
			transfers = cc + transfers
			round_totals = cc + round_totals

		for td in response.css('table tbody tr'):
			name = str(td.css('.candname a::text').extract_first())
			if name=='None':
				name = str(td.css('.candname a b::text').extract_first())
			name = name[0:(name.find('(')-1)]

			yield {
				'election': election[5:],
				'year': year,
				'constit': constit,
				'party' : td.css('td a img ::attr(title)').extract_first(),
				'name' : name,
				'seats': seats,
				'quota': quota,
				'electorate': race_dict['electorate'],
				'turnout': race_dict['turnout'], 
				'valid': race_dict['valid'],
				'spoilt': race_dict['spoilt'],
				'transfers': transfers[i],
				'round_totals': round_totals[i]
			}

			i += 1
		
		next_url = self.driver.find_element_by_xpath("//*[@id='maintablecontent']/div[1]/div[4]/form/div[2]/input[2]")
		next_url.click()
		county = self.driver.find_element_by_xpath('//select[@name="constitid"]//option[@selected]').get_attribute("text")
		if county.strip()==constit.strip():

			next_url = self.driver.find_element_by_xpath('//*[@id="maintablecontent"]/div[1]/div[4]/form/div[2]/select/option[1]')
			next_url.click()
			next_url = self.driver.find_element_by_xpath("//*[@id='maintablecontent']/div[1]/div[4]/form/div[1]/input[1]")
			next_url.click()
		while True:
			try:
				#parsethebodyyourwebdriverhas
				yield scrapy.Request(self.driver.current_url,
					callback = self.parse
					)
			except:
				break	
		self.driver.close()