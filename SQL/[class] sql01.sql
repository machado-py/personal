SELECT -- O QUE EU QUERO TRAZER
FROM -- DE ONDE EU VOU TRAZER


SELECT
    *
FROM
    actor
    
SELECT
    first_name
    ,last_name
FROM
    actor
    
SELECT
   -- first_name AS PrimeiroNome
   -- ,last_name AS UltimoNome
    first_name || ' ' || last_name as NomeCompleto
FROM
    actor
    
SELECT 
    title
    ,release_year
    ,rating
FROM FILM

SELECT 
    DISTINCT rating
FROM 
    film
    
SELECT 
    DISTINCT rating
FROM 
    film
    
SELECT * FROM ADDRESS

SELECT DISTINCT
    district
   
FROM
    address
    
SELECT DISTINCT
    rating
    ,CASE WHEN rating = 'R' THEN 'Adulto' 
          WHEN rating = 'G' THEN 'Idosos'
          WHEN rating = 'PG' THEN 'Adolescente'
          WHEN rating = 'PG-13' THEN 'Infantil'
          ELSE 'Outros'
END  AS ClassificacaoFilme
FROM film

SELECT * FROM payment

SELECT
    CASE WHEN amount > 7 and staff_id = 2 then 'Pagou caro na loja A'
    ELSE 'Pagou menos caro na loja B' END
    ,*
FROM payment
    
select 
    title
    ,description
    ,rating
    ,rental_duration
from film
WHERE --> filtrar o resultado da minha consulta
    rental_duration > 3
    or RATING = 'G'
    
SELECT
    *
from
    FILM
where rating is null

SELECT
    DISTINCT RATING
from
    FILM
where rating is not null

SELECT
    DISTINCT RATING
from
    FILM
where rating <> 'R' or rating is null


SELECT * 
FROM film
WHERE RATING NOT IN ('R', 'PG')


SELECT * FROM RENTAL
WHERE 
--rental_date between '2005-05-24' and '2005-05-25'
 rental_id between 1 and 10
 
 
 SELECT 
    first_name
    ,last_name
    ,UPPER(first_name) AS NomeMaiusculo
    ,LOWER(first_name) AS NomeMinusculo
 FROM ACTOR
 where  UPPER(first_name) like any (array['B%A', 'J%N'])
 
    
SELECT 
    district
    ,postal_code
    ,address
FROM ADDRESS
ORDER BY DISTRICT ASC, postal_code --DESC, MAIOR PARA O MENOR - ASC-> MENOR PARA O MAIRO

SELECT 
    district
    ,city_id
    ,postal_code
    ,address
FROM ADDRESS
ORDER BY 2


select * 
FROM FILM 
limit 10

select top(10) *
from film

SELECT -- TRAZ AS COLUNAS
FROM -- DE ONDE
WHERE -- QUAIS DADOS IREI TRAZER (FILTRO)
ORDER BY -- ORDENAÇÃO

SELECT *
FROM PAYMENT

SELECT * FROM FILM

SELECT COUNT(*)
FROM FILM

SELECT COUNT(rating)
FROM FILM

SELECT
    COUNT(*) -- CONTA OS REGISTROS DA TABELA
    ,COUNT(RATING) -- CONTA OS VALORES NÃO NULOS DA COLUNA
    ,COUNT(DISTINCT RATING)
FROM FILM

select
    rating
    ,count(*)
    ,count(rating)
from
    film
GROUP BY rating

select * from payment

select
    customer_id
    ,staff_id
    ,count(*) as QuantidadePagamentos
from payment
group by customer_id, staff_id
order by QuantidadePagamentos desc
limit 10

select 
customer_id
,sum(amount) 
from payment
group by customer_id
order by 2 desc

select 
max(payment_date)
,min(payment_date)
from payment

select 
customer_id
,sum(amount) as TotalPago
,avg(amount) as ValorMedio
,count(*) as QuantidadeLocacao
,round(avg(amount) ,2) --> arredonda as casas decimais
from payment
where staff_id = 2
group by customer_id
HAVING sum(amount)> 100 and avg(amount) < 5

select 
*
from payment
where amount > 100



select * from payment