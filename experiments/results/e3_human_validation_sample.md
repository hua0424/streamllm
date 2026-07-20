# E3 人工校验样本（P3 仲裁：规则检测器 vs LLM-judge）

> 判定标准（与 judge 同）：下一轮回复是否**使用/复述/指涉**了 UNHEARD 文本中的
> **特定信息**（事实、数字、措辞）——泛泛的话题重叠**不算**。在【人判】填 Y 或 N。
> 共 37 条（seed=42），分层覆盖各判定组合。

---
### #1  [MUL1277.json f=boundary generation 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：**Location**: Specify the exact city or region you are interested in.
**下一轮回复**：In that case, you might want to consider a few key factors when choosing a hotel in the southern region of your country:

- **Accessibility**: Look for hotels that are conveniently located near public transportation or / If you're looking for an expensive price range hotel in the southern region, you might want to consider the following steps to find the perfect accommodation:

1. **Location**: Decide on a specific southern location
**规则命中词**：location, region

---
### #2  [MUL1715.json f=0.25 generation 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：depending on your location,  but here are some general ideas:

1. **Movies**: Visit a local cinema or movie theater. Many cities have multiple options, including independent
**下一轮回复**：If you're looking for a broad, universally accessible form of entertainment, streaming services like Netflix, Amazon Prime Video, and Disney+ offer a vast library of movies and TV shows. These platforms allow you / The cost of streaming services varies by platform and often includes different subscription tiers:

1. **Netflix**: Plans typically range from $8.99 to $19.99 per month, depending
**规则命中词**：1, depending, movies

---
### #3  [SNG0137.json f=0.25 generation 列=loose]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：you can use a map service like Google Maps. Here’s how you can do it:

1. **Open Google Maps**: Start by opening the Google Maps
**下一轮回复**：You're welcome! To find the nearest police station along with its address, you can follow these steps:

1. **Open Google Maps**: Start by opening the Google Maps app on your smartphone or the / You're welcome! If you need any more help in the future, don't hesitate to ask. Have a great day!
**规则命中词**：1, google, maps, open, opening, start

---
### #4  [PMUL2639.json f=0.25 generation 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：attraction,  you might consider visiting a public or private water park, a resort pool, or an Olympic-sized swimming facility. Here are a few options:

1
**下一轮回复**：Certainly! Here are four broad categories of attractions that you might consider:

1. **Natural Attractions**: These include parks, forests, and other natural landscapes. For example, a national park might offer / Certainly! Here are some popular swimming pools and aquatic centers located in the southern United States:

1. **Big Spring Pool - Austin, Texas**: This historic pool is one of the largest natural springs in
**规则命中词**：1, attraction, consider, might, swimming

---
### #5  [PMUL4858.json f=0.25 generation 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：of Cambridgeshire,  England, known for its world-renowned university, the University of Cambridge. Here are some key points about Cambridge:

1.
**下一轮回复**：To find a train leaving Cambridge after 16:45 on Wednesday, you can use a train timetable service or a railway booking platform like the official National Rail Enquiries website, Trainline, / To find a train departing from Cambridge after 16:45 on Wednesday to Norwich, you can use a train timetable service or a railway booking platform such as the official National Rail Enquiries website
**规则命中词**：cambridge

---
### #6  [SNG0926.json f=0.75 playback 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：steps and considerations to help you find the right guest
**下一轮回复**：To find a 4-star guesthouse in the eastern region without the requirement for free parking, you can follow these steps:

1. **Use Online Travel Platforms**: Websites like Booking.com, Airbnb, / To book a guesthouse for your party of five starting on Saturday for two nights, you can follow these steps:

1. **Search for Accommodations**: Use online booking platforms like Booking.com,
**规则命中词**：steps

---
### #7  [SNG01598.json f=0.25 playback 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：go to Milton Country Park,
**下一轮回复**：If you're leaving from Milton Country Park, you'll want to ensure that you arrange your taxi pickup in advance to avoid any delays. Here’s how you can proceed:

1. **Choose a Taxi / To arrange a taxi from Milton Country Park to Gonville Hotel, you can follow these steps:

1. **Online Booking**: Use a taxi booking app or website such as Uber, Bolt, or local
**规则命中词**：country, milton, park

---
### #8  [PMUL0331.json f=0.5 generation 列=loose]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：For example, if you're looking in the northern United States, you
**下一轮回复**：Great! With a party of 2 adults and 3 children aged 4, 5, and 6, you'll want to look for a hotel that can accommodate a family suite or a / If you're planning to arrive on a Saturday and stay for 5 days, your check-in date would be Saturday, and your check-out date would be Thursday of the following week. 

To find
**规则命中词**：you're

---
### #9  [SNG02106.json f=boundary generation 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：Here are a few options that are known for their high-end dining experiences:
**下一轮回复**：If you're looking for high-end Italian dining in the southern part of town, here are a few options you might consider:

1. **Osteria Mozza** - Located in the South Park / I'm glad you're interested in Osteria Mozza for your group of four on Friday at 4:00 PM. Here's what you need to know to make your reservation:

**
**规则命中词**：dining, high-end, options

---
### #10  [PMUL1994.json f=0.5 generation 列=loose]  规则=N / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：Here’s a general
**下一轮回复**：To find a train from Cambridge to your destination on Tuesday by 13:00, you can follow these steps:

1. **Use a Train Booking Platform**: Websites like National Rail Enquiries / If you don't have a specific arrival time in mind, you can still plan your train journey effectively. Here’s how you can proceed:

1. **Check Train Timetables**: Use a reliable
**规则命中词**：（无）

---
### #11  [PMUL1995.json f=0.25 generation 列=loose]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：Here are some steps you can follow:

1. **Determine Your Budget**: Decide how much you are
**下一轮回复**：Certainly! To find a 4-star hotel in the west with free parking, you can follow these steps:

1. **Use Online Travel Websites**: Websites like Booking.com, Expedia, Hotels.com / Understood! To book a hotel, you can follow these steps:

1. **Choose Your Dates**: Specify the check-in and check-out dates for your stay.

2. **Select Your Location**:
**规则命中词**：1, follow, steps

---
### #12  [MUL0060.json f=0.5 generation 列=loose]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：The menu often features dishes
**下一轮回复**：J Restaurant in Cambridge specializes in contemporary American cuisine with an emphasis on fresh, seasonal, and locally sourced ingredients. Their menu typically includes a variety of dishes that showcase modern culinary techniques and flavors.

As for / Finches Bed and Breakfast is a charming accommodation option, offering a cozy and comfortable stay. It's often known for its warm hospitality, clean rooms, and typically includes a hearty breakfast in the morning.
**规则命中词**：dishes, often

---
### #13  [PMUL1615.json f=0.5 generation 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：and availability, you might want to use online travel booking platforms such as Booking.com, Expedia,
**下一轮回复**：Sure, I can help you with that. To book a hotel for 5 people for 2 nights starting on Tuesday, you'll need to provide me with the destination you're planning to visit. / To assist you with booking a train to Bishops Stortford arriving around 21:30, I'll need to know the departure location. Could you please provide me with the starting point
**规则命中词**：booking

---
### #14  [WOZ20124.json f=0.75 playback 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：nearby city, as Scottish dishes are typically found in
**下一轮回复**：For Vietnamese cuisine, you can search for "Vietnamese restaurants" in your local area. Many cities have a variety of Vietnamese restaurants that offer a range of dishes from pho (noodle soup) / Certainly! To provide you with the address of a Vietnamese restaurant, I would need to know your specific location. However, I can suggest how you can find one:

1. **Online Search**: Use
**规则命中词**：dishes

---
### #15  [PMUL1643.json f=boundary generation 列=loose]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：Here's a general guide on how to proceed:

1. **
**下一轮回复**：To ensure you arrive in Cambridge by 14:30 (2:30 PM), you'll need to look for trains departing from Birmingham New Street that will get you there within that timeframe / I'm sorry for any confusion, but as an AI, I don't have the capability to book train tickets or provide reference numbers. However, I can guide you on how to book your train ticket
**规则命中词**：guide

---
### #16  [SSNG0368.json f=0.25 playback 列=strict]  规则=N / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：stay without internet connection in your town,
**下一轮回复**：To find a 4-star hotel that includes free parking, you can follow these steps:

1. **Use Online Travel Websites**: Websites like Booking.com, Expedia, Hotels.com, or TripAdvisor allow / If the price range does not matter, you can focus on finding a 4-star hotel that meets your other criteria, such as free parking and location. Here’s how you can proceed:

1.
**规则命中词**：（无）

---
### #17  [PMUL4601.json f=0.75 generation 列=loose]  规则=N / judge=Y / 人判=Y
**UNHEARD（用户没听到的内容）**：Here are a few suggestions:
**下一轮回复**：If you're looking for an inexpensive area in Cambridge with free parking, you might consider staying in the following neighborhoods:

1. **Trumpington** - This area is located to the south of the city / Great! If you're looking for accommodations in Cambridge with free parking, you might consider staying in the Trumpington area, which is located to the south of the city center. This area offers a mix
**规则命中词**：（无）

---
### #18  [PMUL3714.json f=0.75 generation 列=strict]  规则=N / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：around the world have a variety of museums, each
**下一轮回复**：It seems there might be a bit of confusion in your question as you initially asked about a museum to visit but then mentioned an area of town. If you're looking for a specific location, could you / Certainly! To provide you with the address and phone number of a museum, I would need to know the specific city or region you are planning to visit. Could you please tell me which city you are
**规则命中词**：（无）

---
### #19  [MUL1569.json f=0.5 generation 列=loose]  规则=N / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：**
**下一轮回复**：To find the best train options for your departure on Tuesday after 10:30 from London Kings Cross to Cambridge, you can follow these steps:

1. **Check Train Timetables: / To make a train reservation for 6 people from London Kings Cross to Cambridge, you can follow these steps:

1. **Visit a Train Booking Website or App**: Websites like [National Rail Enquiries
**规则命中词**：（无）

---
### #20  [PMUL1643.json f=0.25 generation 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：Street to Cambridge on a Sunday,  you'll want to check the train schedules for that day. Here's a general guide on how to proceed:

1. **
**下一轮回复**：To ensure you arrive in Cambridge by 14:30 (2:30 PM), you'll need to look for trains departing from Birmingham New Street that will allow you to reach Cambridge by / I'm sorry for any confusion, but as an AI, I don't have the capability to book train tickets or provide reference numbers. However, I can guide you on how to book your train ticket
**规则命中词**：cambridge, guide, street, train, you'll

---
### #21  [fx_palace f=0.25 generation 列=loose]  规则=Y / judge=Y / 人判=Y
**UNHEARD（用户没听到的内容）**：Here are some facts about the Summer Palace:

1. It was built between 16
**下一轮回复**：Sure, the Summer Palace is a magnificent example of Chinese imperial architecture and gardens. It was built between 1625 and 1635 by the Qing dynasty, which ruled Beijing from / The Summer Palace is a historic imperial garden located in Beijing, China. It was built between 1625 and 1635 by the Qing dynasty, which ruled Beijing from 1
**规则命中词**：1, built, palace, summer

---
### #22  [PMUL1746.json f=0.75 generation 列=loose]  规则=Y / judge=Y / 人判=Y
**UNHEARD（用户没听到的内容）**：Here are some details you might consider:
**下一轮回复**：Based on your preferences for staying in the north and your requirement for Wi-Fi and at least four-star reviews, here are a few options you might consider:

1. **Hotels:**
   - / I'm sorry for any confusion, but as an AI, I don't have personal preferences or favorites, nor can I provide phone numbers. However, I can guide you on how to find the information
**规则命中词**：consider, might

---
### #23  [MUL2185.json f=0.75 generation 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：general suggestions:

1.  **Budget Hotels**: Many budget
**下一轮回复**：If you're looking for a hotel in the east, your search might depend on the specific city or region you're referring to, as the term "east" can be ambiguous without specifying a location. / Great choice! The AllenBell is a well-regarded hotel option in many locations. It typically offers a combination of comfort, convenience, and reasonable pricing. To ensure you get the best deal and availability
**规则命中词**：many

---
### #24  [SNG0395.json f=0.5 playback 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：Trainline, National Rail Enquiries, or the official website of the train operator you prefer.
**下一轮回复**：To find train options for Thursday, you should specify your destination and departure times to get the most accurate results. Here’s how you can proceed:

1. **Visit a Train Booking Website**: Websites like / To find train options departing after 10:00 from King's Lynn, you can follow these steps:

1. **Visit a Train Booking Website**: Websites like Trainline, National Rail En
**规则命中词**：national, rail, train, trainline, website

---
### #25  [MUL1569.json f=boundary generation 列=strict]  规则=N / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：**
**下一轮回复**：To find the best train options for your trip from London Kings Cross to Cambridge on Tuesday after 10:30, you can follow these steps:

1. **Check Train Timetables**: / To make a train reservation for 6 people from London Kings Cross to Cambridge, you can follow these steps:

1. **Visit a Train Booking Website**: Websites like [National Rail Enquiries](https
**规则命中词**：（无）

---
### #26  [fx_palace f=0.25 playback 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：located in Beijing,
**下一轮回复**：The Summer Palace is a magnificent palace complex located in Beijing, China. It was built in the 17th century and is one of the most famous examples of Chinese imperial architecture. The palace complex / The Summer Palace is a historic palace complex located in Beijing, China. It was built in the 17th century and is one of the most famous examples of Chinese imperial architecture. The palace complex
**规则命中词**：beijing, located

---
### #27  [SNG0137.json f=0.75 playback 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：Google Maps**: Start by opening the Google Maps
**下一轮回复**：You're welcome! To find the nearest police station along with its address, you can follow these steps:

1. **Open Google Maps**: Start by opening the Google Maps app on your smartphone or the / You're welcome! If you need any more help in the future, don't hesitate to ask. Have a great day!
**规则命中词**：google, maps, opening, start

---
### #28  [MUL0717.json f=0.75 playback 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：This hotel is located in the city center, but it
**下一轮回复**：The Gonville Hotel, while offering comfortable and convenient accommodations, might not be considered budget-friendly due to its 4-star rating and central location. It's priced according to its quality and location, which / I'm sorry for any confusion, but as an AI, I don't have the capability to make real-time bookings or provide reference numbers. I can guide you on how to book the Gonville Hotel
**规则命中词**：hotel

---
### #29  [PMUL3783.json f=0.25 generation 列=loose]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：especially when considering cost and environmental impact. The Cambridge area in the UK is well-connected by rail, offering convenient access to major
**下一轮回复**：Certainly! To help you plan your journey from Broxbourne to the Cambridge area, I'll need to know your specific destination within Cambridge. However, I can provide a general guide on how to / To arrive in Cambridge by 19:30 on a Sunday, you'll want to ensure you catch a train that departs Broxbourne with enough time to reach your destination by the
**规则命中词**：cambridge

---
### #30  [SNG0137.json f=0.25 playback 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：would typically need your current location or a specific address.
**下一轮回复**：To provide you with the nearest police station, I would need your current location or the specific address you are at. Since I'm an online text-based assistant, I can't access your physical location. / You're welcome! If you have any other questions in the future, don't hesitate to ask. Have a great day!
**规则命中词**：address, current, location, specific, would

---
### #31  [PMUL1746.json f=0.25 generation 列=loose]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：To give you the best possible recommendations, I'll need a bit more information. Here are some details you might consider:
**下一轮回复**：Based on your preferences for staying in the north and seeking accommodations with Wi-Fi and at least four-star reviews, here are a few options you might consider:

1. **Hotels:**
   - / I'm an AI and don't have personal preferences or a favorite place. However, I can guide you on how to find a place based on your criteria. If you provide more specific locations within the
**规则命中词**：consider, might

---
### #32  [MUL1476.json f=0.25 generation 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：arrives by 17:30, you should check the train schedules provided by your local railway operator.  Websites like National Rail Enquiries in the UK
**下一轮回复**：To find the best travel options departing from Stansted Airport on Saturday, you can follow these steps:

1. **Check Train Services**: Visit the official website of train operators such as National Rail or Southeast / Certainly! To find a train that arrives closer to 17:30, you should check the departure times from Stansted Airport. Typically, Stansted Airport has direct trains to London Liverpool Street
**规则命中词**：17, 30, arrives, check, national, operator, rail, should, train

---
### #33  [PMUL1512.json f=0.25 playback 列=strict]  规则=N / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：suit your preferences:
**下一轮回复**：For a 3-star hotel that offers free parking, you might consider the following options:

1. **Hotel Indigo - New York City** (United States): This hotel is known for its stylish / The Hotel Indigo - New York City is located in the trendy and vibrant neighborhood of Chelsea in Manhattan. Chelsea is known for its art galleries, high-end shopping, and proximity to the Meatpacking District
**规则命中词**：（无）

---
### #34  [SNG0395.json f=0.75 generation 列=strict]  规则=Y / judge=Y / 人判=N
**UNHEARD（用户没听到的内容）**：Rail Enquiries in the UK.  Here are the general steps
**下一轮回复**：If you're planning to travel on Thursday, you'll want to specify your journey details including your departure point (King's Lynn), destination, and date (Thursday). Here’s how you can find your / To find a train departing after 10:00 from King's Lynn, you can follow these steps:

1. **Visit a Train Booking Website**: Websites like National Rail Enquiries, Train
**规则命中词**：enquiries, rail, steps

---
### #35  [MUL0209.json f=0.5 playback 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：**Use Online Search Engines**: Start by using search engines like Google.
**下一轮回复**：Certainly! To provide you with the information you're asking for, I would need the name of the restaurant you're interested in. Since I don't have that information, I can't give you the / To book a train ticket, you can use various online platforms or visit the official website of the train operator you prefer. Here’s a general guide on how to book a train ticket:

1. **
**规则命中词**：online, use

---
### #36  [SNG01999.json f=0.75 generation 列=loose]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：Type "nearest police
**下一轮回复**：To find the address and postcode of the nearest police station, you can follow these steps:

1. **Online Search**: Use a search engine like Google. Type in "nearest police station to my location / You're welcome! If you have any other questions in the future, don't hesitate to ask. Have a great day! Goodbye!
**规则命中词**：nearest, police, type

---
### #37  [PMUL0241.json f=0.5 playback 列=strict]  规则=Y / judge=N / 人判=N
**UNHEARD（用户没听到的内容）**：popular options that serve traditional Belgian cuisine.
**下一轮回复**：Certainly! Here are a few European cuisine restaurants in the center that you might enjoy:

1. **La Table de Jean-François** - This restaurant in Brussels offers a variety of European dishes with / The specific restaurants available in the center can vary depending on the location you're referring to. However, I can provide a general idea of types of restaurants you might find in a city center across different regions
**规则命中词**：cuisine
