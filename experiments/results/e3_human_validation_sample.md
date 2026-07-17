# E3 人工校验样本（P3 仲裁：规则检测器 vs LLM-judge）

> 判定标准（与 judge 同）：下一轮回复是否**使用/复述/指涉**了 UNHEARD 文本中的
> **特定信息**（事实、数字、措辞）——泛泛的话题重叠**不算**。在【人判】填 Y 或 N。
> 共 23 条（seed=42），分层覆盖各判定组合。

---
### #1  [PMUL1643.json f=boundary generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：Here's a general guide on how to proceed:

1. **
**下一轮回复**：To ensure you arrive in Cambridge by 14:30 (2:30 PM), you'll need to look for trains departing from Birmingham New Street that will get you there within that timeframe / I'm sorry for any confusion, but as an AI, I don't have the capability to book train tickets or provide reference numbers. However, I can guide you on how to book your train ticket
**规则命中词**：guide

---
### #2  [SNG0137.json f=0.25 generation]  规则=Y / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：you can use a map service like Google Maps. Here’s how you can do it:

1. **Open Google Maps**: Start by opening the Google Maps
**下一轮回复**：You're welcome! To find the nearest police station along with its address, you can follow these steps:

1. **Open Google Maps**: Start by opening the Google Maps app on your smartphone or the / You're welcome! If you need any more help in the future, don't hesitate to ask. Have a great day!
**规则命中词**：1, google, maps, open, opening, start

---
### #3  [PMUL0052.json f=0.5 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：Here are a few locations where a Bridge Guest House might
**下一轮回复**：To book the Bridge Guest House for 6 people for 3 nights starting from Thursday, you would typically need to contact the hotel directly or use an online booking platform such as Booking.com, Airbnb, / If you're looking for Chinese food, there are many delicious options you might enjoy. Here are a few popular dishes from different regions of China:

1. **Kung Pao Chicken** - A
**规则命中词**：bridge, guest, house, might

---
### #4  [MUL0460.json f=0.25 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：you can follow these steps:

1. **Check Train Timetables**: Visit the official website of the train operator that runs services between Cambridge and Steven
**下一轮回复**：To find the best train options for your journey from Cambridge to Stevenage on Monday that will have you arriving by 16:30, you should check the train schedules directly with a reputable train / To make a booking for 7 people on a train from Cambridge to Stevenage, you would typically need to follow these steps:

1. **Check Train Schedules**: First, ensure you have the
**规则命中词**：1, cambridge, check, follow, steps, steven, train

---
### #5  [PMUL4601.json f=0.75 generation]  规则=N / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：Here are a few suggestions:
**下一轮回复**：If you're looking for an inexpensive area in Cambridge with free parking, you might consider staying in the following neighborhoods:

1. **Trumpington** - This area is located to the south of the city / Great! If you're looking for accommodations in Cambridge with free parking, you might consider staying in the Trumpington area, which is located to the south of the city center. This area offers a mix
**规则命中词**：（无）

---
### #6  [PMUL4864.json f=boundary generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：In many rural areas or smaller towns, it
**下一轮回复**：Certainly! Many 2-star guesthouses around the world offer free parking to their guests. This can be found in various locations, from countryside retreats to smaller towns and cities. To ensure you find / Certainly! Many 3-star hotels and guesthouses around the world provide free parking to their guests. This can be a great option if you're traveling with a vehicle. You can find these accommodations in
**规则命中词**：smaller, towns

---
### #7  [SNG01999.json f=0.75 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：Type "nearest police
**下一轮回复**：To find the address and postcode of the nearest police station, you can follow these steps:

1. **Online Search**: Use a search engine like Google. Type in "nearest police station to my location / You're welcome! If you have any other questions in the future, don't hesitate to ask. Have a great day! Goodbye!
**规则命中词**：nearest, police, type

---
### #8  [MUL1715.json f=boundary generation]  规则=N / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：**Theaters**: Visit a local theater for plays, musical
**下一轮回复**：If you're looking for entertainment without specifying a location, here are some universally accessible options:

1. **Streaming Services**: Platforms like Netflix, Amazon Prime, Hulu, and Disney+ offer a wide range / I'm sorry for any confusion, but your question seems to be a bit mixed. You mentioned "cost" and "postal code," but didn't specify what you're asking the cost of or the
**规则命中词**：（无）

---
### #9  [SSNG0368.json f=0.25 generation]  规则=Y / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：you can follow these steps:

1. **Use Travel Websites**: Websites like Booking.com, TripAdvisor, or Expedia allow
**下一轮回复**：Yes, many travel websites like Booking.com, TripAdvisor, and Expedia allow you to filter your search results based on various amenities, including free parking. Here’s how you can do it on each platform / If the price range does not matter to you, you can still use the following steps to find a 4-star hotel with free parking:

1. **Search on Travel Websites**: Use websites like Booking
**规则命中词**：1, allow, booking, expedia, follow, steps, travel, tripadvisor, use, websites

---
### #10  [PMUL0331.json f=0.5 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：For example, if you're looking in the northern United States, you
**下一轮回复**：Great! With a party of 2 adults and 3 children aged 4, 5, and 6, you'll want to look for a hotel that can accommodate a family suite or a / If you're planning to arrive on a Saturday and stay for 5 days, your check-in date would be Saturday, and your check-out date would be Thursday of the following week. 

To find
**规则命中词**：you're

---
### #11  [SNG01319.json f=0.25 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：Could you please provide your city or a specific address? Alternatively, you can
**下一轮回复**：Certainly! To provide you with the most accurate information, I would need to know your current location. However, you can generally find the phone numbers for hospitals by searching online for hospitals in your area. / I'm unable to provide real-time information or call on your behalf. However, I can guide you on how to find the address of a hospital. You can search for hospitals in your area using online
**规则命中词**：address, provide

---
### #12  [WOZ20119.json f=0.5 generation]  规则=N / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：However, I can
**下一轮回复**：I'm sorry for any confusion, but as an AI, I don't have real-time access to specific databases or the internet to look up current phone numbers for businesses. I suggest searching online for the / You're welcome! Don't hesitate to return if you have more questions in the future. Have a great day!
**规则命中词**：（无）

---
### #13  [fx_palace f=0.25 generation]  规则=Y / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：Here are some facts about the Summer Palace:

1. It was built between 16
**下一轮回复**：Sure, the Summer Palace is a magnificent example of Chinese imperial architecture and gardens. It was built between 1625 and 1635 by the Qing dynasty, which ruled Beijing from / The Summer Palace is a historic imperial garden located in Beijing, China. It was built between 1625 and 1635 by the Qing dynasty, which ruled Beijing from 1
**规则命中词**：1, built, palace, summer

---
### #14  [MUL0060.json f=0.5 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：The menu often features dishes
**下一轮回复**：J Restaurant in Cambridge specializes in contemporary American cuisine with an emphasis on fresh, seasonal, and locally sourced ingredients. Their menu typically includes a variety of dishes that showcase modern culinary techniques and flavors.

As for / Finches Bed and Breakfast is a charming accommodation option, offering a cozy and comfortable stay. It's often known for its warm hospitality, clean rooms, and typically includes a hearty breakfast in the morning.
**规则命中词**：dishes, often

---
### #15  [PMUL1994.json f=0.5 generation]  规则=N / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：Here’s a general
**下一轮回复**：To find a train from Cambridge to your destination on Tuesday by 13:00, you can follow these steps:

1. **Use a Train Booking Platform**: Websites like National Rail Enquiries / If you don't have a specific arrival time in mind, you can still plan your train journey effectively. Here’s how you can proceed:

1. **Check Train Timetables**: Use a reliable
**规则命中词**：（无）

---
### #16  [PMUL3783.json f=0.25 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：especially when considering cost and environmental impact. The Cambridge area in the UK is well-connected by rail, offering convenient access to major
**下一轮回复**：Certainly! To help you plan your journey from Broxbourne to the Cambridge area, I'll need to know your specific destination within Cambridge. However, I can provide a general guide on how to / To arrive in Cambridge by 19:30 on a Sunday, you'll want to ensure you catch a train that departs Broxbourne with enough time to reach your destination by the
**规则命中词**：cambridge

---
### #17  [PMUL1746.json f=0.75 generation]  规则=Y / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：Here are some details you might consider:
**下一轮回复**：Based on your preferences for staying in the north and your requirement for Wi-Fi and at least four-star reviews, here are a few options you might consider:

1. **Hotels:**
   - / I'm sorry for any confusion, but as an AI, I don't have personal preferences or favorites, nor can I provide phone numbers. However, I can guide you on how to find the information
**规则命中词**：consider, might

---
### #18  [MUL0209.json f=0.25 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：you can follow these steps:

1. **Use Online Search Engines**: Start by using search engines like Google. Type in "expensive North American
**下一轮回复**：I'm sorry for any confusion, but as an AI, I don't have real-time access to specific databases or the ability to browse the internet in real-time to provide current information such as the location / To book a train ticket, you can use various online platforms or visit the official website of train operators such as National Rail Enquiries in the UK, or Amtrak for train travel in the United States
**规则命中词**：online, use

---
### #19  [MUL1715.json f=0.5 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：**Movies**: Visit a local cinema or movie theater. Many cities have multiple options, including art
**下一轮回复**：Sure, here are some general options for entertainment that you might consider:

1. **Movies**: Visit a local cinema or movie theater. Many cities have multiple options, including art house theaters, multiplex / I'm sorry for any confusion, but I can't provide specific information about costs or postal codes for entertainment venues as those details can vary greatly depending on your location. However, I can give you a
**规则命中词**：cinema, cities, including, local, many, movie, movies, multiple, options, theater, visit

---
### #20  [PMUL1746.json f=0.25 generation]  规则=Y / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：To give you the best possible recommendations, I'll need a bit more information. Here are some details you might consider:
**下一轮回复**：Based on your preferences for staying in the north and seeking accommodations with Wi-Fi and at least four-star reviews, here are a few options you might consider:

1. **Hotels:**
   - / I'm an AI and don't have personal preferences or a favorite place. However, I can guide you on how to find a place based on your criteria. If you provide more specific locations within the
**规则命中词**：consider, might

---
### #21  [SNG02106.json f=0.25 generation]  规则=Y / judge=Y / 人判=____
**UNHEARD（用户没听到的内容）**：you might consider some of the world's most renowned Italian restaurants. Here are a few options that are known for their high-end dining experiences:
**下一轮回复**：If you're looking for high-end Italian dining in the southern part of town, here are a few options you might consider:

1. **Osteria Mozza** - Located in the South Park / I'm glad you're interested in Osteria Mozza for your group of four on Friday at 4:00 PM. Here's what you need to know:

**Osteria Mo
**规则命中词**：consider, dining, high-end, italian, might, options

---
### #22  [MUL1569.json f=0.5 generation]  规则=N / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：**
**下一轮回复**：To find the best train options for your departure on Tuesday after 10:30 from London Kings Cross to Cambridge, you can follow these steps:

1. **Check Train Timetables: / To make a train reservation for 6 people from London Kings Cross to Cambridge, you can follow these steps:

1. **Visit a Train Booking Website or App**: Websites like [National Rail Enquiries
**规则命中词**：（无）

---
### #23  [PMUL1995.json f=0.25 generation]  规则=Y / judge=N / 人判=____
**UNHEARD（用户没听到的内容）**：Here are some steps you can follow:

1. **Determine Your Budget**: Decide how much you are
**下一轮回复**：Certainly! To find a 4-star hotel in the west with free parking, you can follow these steps:

1. **Use Online Travel Websites**: Websites like Booking.com, Expedia, Hotels.com / Understood! To book a hotel, you can follow these steps:

1. **Choose Your Dates**: Specify the check-in and check-out dates for your stay.

2. **Select Your Location**:
**规则命中词**：1, follow, steps
