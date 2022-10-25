SELECT * FROM orders;
SELECT * FROM order_details;
SELECT * FROM products;
SELECT * FROM customers;

-- 1 Obtenha uma tabela que contenha o id do pedido 
-- e o valor total do mesmo.
select 
	orders.order_id
	,(order_details.unit_price * order_details.quantity) - order_details.discount
from orders
left join order_details on orders.order_id = order_details.order_id
order by 2 desc;

-- 2 Obtenha uma lista dos 10 clientes que realizaram
-- o maior número de pedidos, bem como o número de pedidos de cada,
-- ordenados em ordem decrescente de nº de pedidos.
select
	customer_id
	,count(order_id)
from orders
group by 1
order by 2 desc
limit 10;

-- 3 Obtenha uma tabela que contenha o id e o valor total do pedido
-- e o nome do cliente que o realizou.
select 
	orders.order_id
	,customers.company_name
	,(order_details.unit_price * order_details.quantity) - order_details.discount
from orders
left join order_details on orders.order_id = order_details.order_id
left join customers on orders.customer_id = customers.customer_id
order by 3 desc;

-- 4 Obtenha uma tabela que contenha o país do cliente e o valor da
-- compra que ele realizou.
select 
	customers.company_name
	,customers.country
	,(order_details.unit_price * order_details.quantity) - order_details.discount
from orders
left join order_details on orders.order_id = order_details.order_id
left join customers on orders.customer_id = customers.customer_id
order by 3 desc;

-- 5 Obtenha uma tabela que contenha uma lista dos países dos 
-- clientes e o valor total de compras realizadas em cada um
-- dos países. Ordene a tabela, na order descrescente, considerando 
-- o valor total de compras realizadas por país.
select
	country
	,sum(total)
from(
	select 
		customers.country
		,(order_details.unit_price * order_details.quantity) - order_details.discount as total
	from orders
	left join order_details on orders.order_id = order_details.order_id
	left join customers on orders.customer_id = customers.customer_id)
as base
group by country
order by 2 desc;

-- 6 Obtenha uma tabela com o valor médio das vendas em cada mês
-- (ordenados do mês com mais vendas para o mês com menos vendas).
select
	date_trunc('month', order_date) as year_month
	,avg(total) as average_total
from(
	select 
		orders.order_date
		,(order_details.unit_price * order_details.quantity) - order_details.discount as total
	from orders
	left join order_details on orders.order_id = order_details.order_id)
as base
group by 1
order by 1 asc;

