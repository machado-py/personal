-- Exercício 1 -------------------------------------
select *
from triangulos

select
	*
	,case 
		when (a+b)<=c then 'valores não formam um triângulo'
		when a = b and b = c then '3 lados iguais'
		when (a = b and a <> c) or (a=c and a<>b) then '2 lados iguais'
		when a <> b and a <> c then '3 lados diferentes'
	end as resultado2
from triangulos

-- Exercício 2 -------------------------------------
select *
from flavours

select
	category
	,count(category)
from flavours
where category <> 'NULL'
group by category
order by 2 desc

-- Exercício 3 -------------------------------------
select
	'A categoria ' || category || ' possui ' || count(*) as texto
from flavours
where category <> 'NULL'
group by category
order by count(*) desc
