import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from prompt_toolkit import prompt as prompt_input


class GoogleSearchScraper:
    def scrape_google_links(search_query, num_links=10):
        """
        Fetches the link URLs based on the search query
        """
        
        # Set Chrome options for headless mode
        options = Options()
        options.add_argument("start-maximized")
        options.add_argument("--headless")
        options.add_argument("--disable-notifications")  # to disable notifications

        # Initialize a Chrome driver
        driver = uc.Chrome(options=options)

        links = []
        i = 0

        while len(links) < num_links:
            # Format the search query
            formatted_query = search_query.replace(' ', '+')

            # Perform a Google search
            url = f"https://www.google.com/search?q={formatted_query}&start={i * 10}"
            driver.get(url)

            # Wait until the search results are loaded
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "g"))
            )

            # Extract the URLs of the search results
            search_results = driver.find_elements(By.CLASS_NAME, 'g')
            for result in search_results:
                if len(links) >= num_links:
                    break
                link_element = result.find_element(By.CSS_SELECTOR, 'a')
                link = link_element.get_attribute('href')
                links.append(link)

            i += 1

        # Close the driver
        driver.quit()

        return links



    def extract_page_info(link):
        # Set Chrome options for headless mode
        options = Options()
        options.add_argument("--headless")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_argument("--disable-notifications")  # to disable notifications

        # Initialize a Chrome driver
        driver = webdriver.Chrome(options=options)

        # Navigate to the page
        driver.get(link)

        # Extract the text from the body of the page
        body_text = driver.find_element(By.TAG_NAME, 'body').text

        # Close the driver
        driver.quit()

        return body_text
