# Demo shop database

A small e-commerce database. Dates are stored as text in ISO format (YYYY-MM-DD), so
`strftime('%Y', order_date)` gives the year and `substr(order_date, 1, 7)` gives the month.

## Table customers

One row per customer.

- `customer_id` (integer, primary key): unique customer number.
- `name` (text): full name, first name then last name.
- `email` (text): email address.
- `country` (text): country of residence. Values: France, Germany, Spain, Italy, Belgium, Portugal.
- `signup_date` (text): date the customer created an account, between 2022 and 2023.

## Table products

One row per product in the catalogue (25 products).

- `product_id` (integer, primary key): unique product number.
- `name` (text): product name.
- `category` (text): one of Electronics, Sports, Books, Home, Clothing.
- `unit_price` (real): current list price in euros.

## Table orders

One row per order. An order belongs to one customer and contains one or more order items.

- `order_id` (integer, primary key): unique order number.
- `customer_id` (integer): foreign key to `customers.customer_id`.
- `order_date` (text): date the order was placed, between 2022 and 2024.
- `status` (text): one of delivered, shipped, pending, cancelled.

## Table order_items

One row per product line inside an order.

- `order_item_id` (integer, primary key): unique line number.
- `order_id` (integer): foreign key to `orders.order_id`.
- `product_id` (integer): foreign key to `products.product_id`.
- `quantity` (integer): number of units bought, 1 to 3.
- `unit_price` (real): price per unit in euros at the time of the order.

## Business definitions

- Line amount = `order_items.quantity * order_items.unit_price`.
- Revenue means the sum of line amounts over orders whose status is not `cancelled`.
  Cancelled orders never count as revenue.
- Order value means the sum of the line amounts of one order.
- A customer is active in a year if they placed at least one order in that year.
- To link customers to products, join `customers` -> `orders` -> `order_items` -> `products`.
