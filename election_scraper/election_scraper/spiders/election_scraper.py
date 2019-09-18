import scrapy
import ast
import re
from selenium import webdriver

class ElectionSpider(scrapy.Spider):

    name = 'electionspider' 
    
    def __init__(self, \
                 start_urls=["http://irelandelection.com/election.php?elecid=157&constitid=109&electype=5"], \
                 local = False):
        
        if type(start_urls)!=list:
            start_urls = [start_urls]
            
        self.start_urls = start_urls
        self.local = bool(local)
        self.driver = webdriver.PhantomJS()
        self.driver.get(self.start_urls[0])

    def parse(self, response):
        
        
        
        try:
            seats = int(response.css('.well table tr td::text').extract()[0][-1])
            quota = re.search('\d+', str(response.css('.well table tr td::text'). \
                            extract()[1]))
            quota = int(quota.group(0))
            election = response.xpath('//select[@name="elecid"]//option[@selected]/text()') \
                .extract_first()
            
            if not self.local:    
                county = None
                constit = response.xpath('//select[@name="constitid"]//option[@selected]/text()').extract_first()
            else:
                county = ' '.join(response.xpath('//*[@id="maintablecontent"]/div[1]/div[2]/h1/a[1]/text()') \
                    .extract_first().split(' ')[:-2]).strip()
                    
                constit = response.xpath("(//select[@name='constitid']//option[@selected]/text())[last()]") \
                    .extract_first().strip()
    
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
                    'county': county,
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
        except:
            pass

        try:
            next_url=self.driver.find_element_by_xpath("(//select[@name='constitid']//option[@selected])[last()]/following-sibling::option[1]")
            next_url.click()
        except:            
            next_url = self.driver.find_element_by_xpath("//*[@id='maintablecontent']/div[1]/div[4]/form/div[1]/input[1]")
            next_url.click()

            next_url = self.driver.find_element_by_xpath('//*[@id="maintablecontent"]/div[1]/div[4]/form/div[2]/select/option[1]')
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