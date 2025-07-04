from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import handle_upload_and_generate
from .gen import handle_upload_and_generate


@api_view(['POST'])
def generate_data(request):
    uploaded_file = request.FILES.get("file")
    num_rows = int(request.data.get("num_rows", 100))
    output_file_type = request.POST.get("output_file_type")  # <-- ADD THIS LINE
    model_type = "ctgan"
    if not uploaded_file:
        return Response({"error": "No file uploaded."}, status=400)
    try:
        file_url = handle_upload_and_generate(uploaded_file, num_rows, model_type,
                                              output_file_type)
        return Response({"status": "success", "file": file_url})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    

@api_view(['POST'])
def generate_smart_data(request):
    uploaded_file = request.FILES.get("file")
    num_rows = int(request.data.get("num_rows", 100))
    output_file_type = request.POST.get("output_file_type")  # <-- ADD THIS LINE
    model_type = "ctgan"
    print("here...")
    if not uploaded_file:
        return Response({"error": "No file uploaded."}, status=400)
    try:
        file_url = handle_upload_and_generate(uploaded_file, num_rows, model_type,
                                              output_file_type)
        print(file_url)
        return Response({"status": "success", "file": file_url})
    except Exception as e:
        print("error",e)
        return Response({"error": str(e)}, status=500)
