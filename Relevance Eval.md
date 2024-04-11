```python
System:
You are an Marketing Content Generation Expert. You will be given the definition of an evaluation metric for assessing the quality of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the provided evaluation metric.

User:
Relevance measures how well the answer addresses the main aspects of the question, based on the context, reference to popular Mainland China terminology or local points of interest, and avoidance of words that may be interpretted by Mainland China Users as offensive or risky. Consider whether all and only the important aspects are contained in the answer when evaluating relevance. Given the context and question, score the relevance of the answer between one to five stars using the following rating scale:
One star: the answer completely lacks relevance
Two stars: the answer mostly lacks relevance
Three stars: the answer is partially relevant
Four stars: the answer is mostly relevant
Five stars: the answer has perfect relevance

This rating value should always be an integer between 1 and 5. So the rating produced should be 1 or 2 or 3 or 4 or 5.

context: Neb Cafe is a luxury coffee company that offers espresso and regular coffee products at affordable prices to customers in Mainland China. Our company's brand is closely tied with high status and family oriented.
question: Create an advertisement.
answer: Indulge in the luxury of Neb Cafe's premium lattes without stepping out of your cozy abode.
stars: 1

context: Neb Cafe is a luxury coffee company that offers espresso and regular coffee products at affordable prices to customers in Mainland China. Our company's brand is closely tied with high status and family oriented.
question: Please create a Neb Cafe latte advertisement for a Blog post.
answer: 🌟 \"Elevate Every Morning\" ☕️\n\nIndulge in the luxury of Neb Cafe's premium lattes without stepping out of your cozy abode. Our carefully crafted espresso meets velvety milk, creating the perfect cup that turns your daily coffee ritual into an affair of elegance and comfort. Discover the richness of our carefully sourced beans and the creamy finish that lingers with every sip, all at an affordable price.
stars: 2

context: Neb Cafe is a luxury coffee company that offers espresso and regular coffee products at affordable prices to customers in Mainland China. Our company's brand is closely tied with high status and family oriented.
question: Please create a Neb Cafe latte advertisement for a Blog post with the goal to Boost brand awareness.
answer: 🌟 \"Elevate Every Morning\" ☕️\n\nIndulge in the luxury of Neb Cafe's premium lattes without stepping out of your cozy abode. Our carefully crafted espresso meets velvety milk, creating the perfect cup that turns your daily coffee ritual into an affair of elegance and comfort. Discover the richness of our carefully sourced beans and the creamy finish that lingers with every sip, all at an affordable price.\n\nEnvision a serene Shanghai morning, the Pearl Tower glinting in the distance, enveloped by the comforting aroma of a freshly brewed Neb Cafe latte.
stars: 3

context: Neb Cafe is a luxury coffee company that offers espresso and regular coffee products at affordable prices to customers in Mainland China. Our company's brand is closely tied with high status and family oriented.
question: Please create a Neb Cafe latte advertisement for a Blog post with the goal to Boost brand awareness for the following audience We are targeting our ad to heads of household adults.
answer: 🌟 \"Elevate Every Morning\" ☕️\n\nIndulge in the luxury of Neb Cafe's premium lattes without stepping out of your cozy abode. Our carefully crafted espresso meets velvety milk, creating the perfect cup that turns your daily coffee ritual into an affair of elegance and comfort. Discover the richness of our carefully sourced beans and the creamy finish that lingers with every sip, all at an affordable price.\n\nEnvision a serene Shanghai morning, the Pearl Tower glinting in the distance, enveloped by the comforting aroma of a freshly brewed Neb Cafe latte. It's more than just caffeine; it's a moment of tranquility before the bustling day begins. It's a tradition that harmonizes with the rhythm of your household—a shared pause to savor and connect.
stars: 4

context: Neb Cafe is a luxury coffee company that offers espresso and regular coffee products at affordable prices to customers in Mainland China. Our company's brand is closely tied with high status and family oriented.
question: Please create a Neb Cafe latte advertisement for a Blog post with the goal to Boost brand awareness for the following audience We are targeting our ad to heads of household adults who prefer to make their coffee at home and coffee is an essential part of their morning routine.
answer: 🌟 \"Elevate Every Morning\" ☕️\n\nIndulge in the luxury of Neb Cafe's premium lattes without stepping out of your cozy abode. Our carefully crafted espresso meets velvety milk, creating the perfect cup that turns your daily coffee ritual into an affair of elegance and comfort. Discover the richness of our carefully sourced beans and the creamy finish that lingers with every sip, all at an affordable price.\n\nEnvision a serene Shanghai morning, the Pearl Tower glinting in the distance, enveloped by the comforting aroma of a freshly brewed Neb Cafe latte. It's more than just caffeine; it's a moment of tranquility before the bustling day begins. It's a tradition that harmonizes with the rhythm of your household—a shared pause to savor and connect.\n\nStep into the world of Neb Cafe. Visit our website to uncover your next at-home coffee favorite and join our community of connoisseurs on social media. Start your day with sophistication. Start your day with Neb Cafe.\n\n🔗 [Explore Neb Cafe](#) 🌐\n👍 Follow us on WeChat and Weibo for more heartwarming stories and exclusive offers!
stars: 5

context: {{context}}
question: {{question}}
answer: {{answer}}
stars:
