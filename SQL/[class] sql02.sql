select DISTINCT RATING from film

SELECT  * FROM FILM LIMIT 10

selecT
    DISTINCT
        rating
        ,case when cast(rating as varchar(10)) is null then 'Sem Classificação' else cast(rating as varchar(10)) end as Classificacao
        ,COALESCE(CAST(RATING AS VARCHAR(10)), 'Sem Classificação') as Classificacao2
FROM
    film
    
SELECT * FROM ADDRESS

SELECT DISTINCT address2, COALESCE(ADDRESS2, 'Sem complemento') as Complemento FROM ADDRESS
SELECT DISTINCT address2, NULLIF(address2,''), coalesce(NULLIF(address2,''),'Sem Complemento') from address



NULLIF(NULLIF(provider,'999999'),'0000000')


SELECT 
    CAST('14/11/1985' AS VARCHAR(10)) AS Aniversario
    ,TO_DATE(CAST('14/11/1985' AS VARCHAR(10)), 'DD/MM/YYYY') AS AniversarioFormatado
    ,TO_DATE(CAST('14-11/1985' AS VARCHAR(10)), 'DD-MM/YYYY') AS AniversarioFormatado2
    , 'Teodor' as pessoa
    
select 
date_part('year', payment_date) as ANO
,date_part('month', payment_date) as MES
,date_part('day', payment_date) as DIA
,payment_date
,year(payment_date)
,* 
from payment

SELECT * FROM CITY
SELECT * FROM COUNTRY

PK -> PRIMARY_KEY (CHAVE PRIMARIA)

select * from country order by country_id desc
INSERT INTO COUNTRY VALUES (112, 'Lua', '2022-07-28')

select * from city
FK -> FOREIGN KEY

Tabela  - Campo         - Descricao
Country - Country_id    - Chave Primária (PK)
City    - Country_ID    - Chave Estrangeira(FK)

select
    --nome da cidade
    city
    --nome do pais
    ,country
    ,country.COUNTRY_ID
FROM 
    city
INNER JOIN country on city.country_id = country.country_id --> os valores devem existir em ambas as tabelas
order by 3 desc

select * from country where country_id = 110
select * from city where country_id = 110

select * from city

insert into city values (602, 'Contagem', 300, '2022-07-28')

select
 --nome do pais
    country AS NomePais
    --nome da cidade
    ,city   
    ,city_id
    ,a.COUNTRY_ID
    ,a.LAST_update
FROM 
    country AS a
LEFT JOIN city as b on b.country_id = a.country_id --> os valores devem existir em ambas as tabelas
order by 3 desc

SELECT
    a.country_id
    ,country
    ,count(*) as ContaTudo --- contando todos os registros retornados
    ,count(b.*) as ContagemCidades
    ,count(b.country_id) as ContagemCidades2
    ,count(a.*) as ContagemEstados
FROM
    country a
    left join city b on a.country_id = b.country_id
GROUP BY a.country_Id, COUNTRY
order by 1 desc


select
 --nome do pais
    country AS NomePais
    --nome da cidade
    ,city   
    ,city_id
    ,a.COUNTRY_ID
    ,a.LAST_update
FROM 
    country AS a
LEFT JOIN city as b on b.country_id = a.country_id --> os valores devem existir em ambas as tabelas
where b.country_id is null
order by 3 desc


select
 --nome do pais
    country AS NomePais
    --nome da cidade
    ,city   
    ,city_id
    ,a.COUNTRY_ID
    ,a.LAST_update
FROM 
    country AS a
RIGHT JOIN city as b on b.country_id = a.country_id --> os valores devem existir em ambas as tabelas

order by 3 desc

create view DadosClientes AS
select 
    first_name
    ,last_name
    ,address
    ,district
    ,c.city_id
    ,city
    ,country
from customer AS a
inner join address AS b on a.address_id = b.address_id
inner join city as c on b.city_id = c.city_id
inner join country as d on c.country_id = d.country_id


select * from film_list

select * from DadosClientes


SELECT 'Consulta 1' as Texto, 1 as Numero
--UNION --> TRAZ OS VALORES E REMOVE AS DUPLICADAS
UNION ALL --> TRAZ OS VALORES, MESMO QUE REPITA
SELECT 'Consulta 1' as Texto, 1 as Numero


SELECT AVG(AMOUNT) FROM PAYMENT

explain analyse
SELECT * 
,(SELECT AVG(AMOUNT) FROM PAYMENT) AS Media
FROM PAYMENT
WHERE AMOUNT >= (SELECT AVG(AMOUNT) FROM PAYMENT)

SELECT
    staff_id
    ,avg(amount)
FROM 
    payment
GROUP BY staff_id

explain analyse
SELECT
    a.staff_id
    ,a.amount
    ,(select avg(b.amount) from payment b where a.staff_id = b.staff_id group by b.staff_id)
FROM
    payment as a
WHERE AMOUNT >= (select avg(b.amount) from payment b where a.staff_id = b.staff_id group by b.staff_id)

select * from payment


WITH ComandoCTE AS (
    select 
        staff_id AS Funcionario
        ,avg(amount) AS ValorMedio
    from payment b 
    group by staff_id
)
select * 
from 
    payment a
    inner join ComandoCTE as b on a.staff_id = b.Funcionario
WHERE
    a.amount >= b.ValorMedio
    
WITH MediaCustoCTE AS 
(SELECT 
    rating
    ,avg(rental_rate) as MediaCusto
FROM FILM
group by rating)
select 
    a.title
    ,a.RATING
    ,rental_rate
    ,MediaCusto
    ,round(rental_rate/MediaCusto,2) * 100 as Proporcao
FROM FILM as A
inner join MediaCustoCTE as B on a.rating = b.rating


select * from MediaCustoCTE


select * from film