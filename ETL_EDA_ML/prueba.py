# Dejo aca el codigo que utilice para solicitar el EDA
# 
# "/dataset_info?page={pagina}&page_size=10": "Endpoint de prueba para revisar el dataset desde el 0 hasta el 453, con un tamaño de 10",
# 
# 
# @app.get("/dataset_info")
# def dataset_info(skip: int = Query(0, alias="page", ge=0), limit: int = Query(1000, le=1000)):
#    """
#    Devuelve un subconjunto del dataset.
#
#    - skip: número de la página para saltar (por defecto 0)
#    - limit: número de registros por página (por defecto 1000, máximo 1000)
#    """
#    try:
#        # Seleccionar la página de datos
#        start = skip * limit
#        end = start + limit
#
#        # Asegurarse de que no se superen los límites del DataFrame
#        if start >= len(df):
#            raise HTTPException(status_code=404, detail="No hay más datos para mostrar.")
#
#        # Extraer el subconjunto de datos
#        subset = df.iloc[start:end].replace({np.nan: None, np.inf: None, -np.inf: None})
#
#        return {"columns": df.columns.tolist(), "data": subset.to_dict(orient="records")}
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))